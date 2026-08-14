# -*- coding: utf-8 -*-
"""
step3_threshold_analysis.py

목적
    solo 이미지(튜브 1개)의 형광 절댓값 분포로부터 음성 기준선(cutoff)을 산출한다.
    train 음성 데이터의 분포를 기준으로 삼고, test 데이터로 그 기준을 평가한다.

방법
    1. YOLO 로 ROI 검출 (conf=0.70 고정)
    2. ROI 의 G채널 95백분위수를 이미지 대표값 I 로 사용
       (ROI 가 여러 개면 그중 최댓값)
    3. train/neg 분포에 Shapiro-Wilk 정규성 검정
       - 정규분포로 볼 수 있으면  cutoff = mean + 3*SD
       - 아니면                   cutoff = 99.7 백분위수
    4. test 데이터에 cutoff 를 적용해 혼동행렬 산출

출력
    results/step3_threshold/
      ├── solo_values.csv            이미지별 I 값
      ├── neg_train_values.csv       train 음성 값만
      ├── neg_baseline_stats.json    분포 통계 + cutoff
      ├── neg_baseline_qq.png        Q-Q plot
      ├── summary.json               전체 요약
      └── viz/                       (--save_viz 지정 시에만)

실행
    python analysis/step3_threshold_analysis.py

    # 검출 결과 이미지도 저장하려면 (용량 주의)
    python analysis/step3_threshold_analysis.py --save_viz

주의
    train 음성 n=40 에서 99.7 백분위수를 쓰면 사실상 최댓값에 가까워지므로,
    극단값 하나가 cutoff 전체를 좌우한다. 이 한계는 README 에 기술되어 있다.

수정 이력
    - 하드코딩 경로를 paths.py 로 이관
    - 오버레이 이미지 저장을 기본 끄기로 변경 (PNG 누적 용량 문제)
    - 저장 시에도 JPEG + 리사이즈로 용량 축소
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")            # 화면 없는 환경에서도 저장되도록
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# ---- 저장소 루트를 import 경로에 추가 (analysis/ 하위에서 실행되므로) ----
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as P

try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit("ultralytics 가 필요합니다:  pip install ultralytics") from e


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 오버레이 저장 시 설정
VIZ_MAX_WIDTH = 1024             # 이보다 크면 줄여서 저장
VIZ_JPEG_QUALITY = 85


# ==================================================================
# 유틸
# ==================================================================
def list_images(root: Path):
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTS)


def imread_unicode(path: Path):
    """한글 경로에서도 안전하게 읽는다."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def solo_label_from_path(path: Path):
    """경로에 pos / neg 폴더가 있는지로 라벨을 정한다."""
    parts = {p.lower() for p in path.parts}
    if "pos" in parts:
        return "pos"
    if "neg" in parts:
        return "neg"
    return None


def safe_crop(img, xyxy):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    H, W = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def g_p95_intensity(crop_bgr) -> float:
    """ROI 의 G채널 95백분위수. 형광 대표값."""
    if crop_bgr is None:
        return np.nan
    G = crop_bgr[:, :, 1].astype(np.float32)
    return float(np.percentile(G, 95.0))


def draw_box_with_label(img, xyxy, color, label, thickness=4):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)


def save_viz(img, out_path: Path):
    """JPEG 로 축소 저장. PNG 원본으로 쌓으면 용량이 급격히 커진다."""
    h, w = img.shape[:2]
    if w > VIZ_MAX_WIDTH:
        scale = VIZ_MAX_WIDTH / w
        img = cv2.resize(img, (VIZ_MAX_WIDTH, int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path.with_suffix(".jpg")), img,
                [cv2.IMWRITE_JPEG_QUALITY, VIZ_JPEG_QUALITY])


def qq_plot(values, out_png: Path, title: str):
    if len(values) < 3:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    stats.probplot(values, dist="norm", plot=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def eval_cutoff(pos_vals, neg_vals, cutoff):
    TP = sum(1 for v in pos_vals if v >= cutoff)
    FN = sum(1 for v in pos_vals if v < cutoff)
    FP = sum(1 for v in neg_vals if v >= cutoff)
    TN = sum(1 for v in neg_vals if v < cutoff)
    total = max(1, len(pos_vals) + len(neg_vals))
    prec = TP / max(1, TP + FP)
    rec = TP / max(1, TP + FN)
    return {
        "TP": TP, "FN": FN, "FP": FP, "TN": TN,
        "ACC": (TP + TN) / total,
        "PREC": prec,
        "RECALL": rec,
        "F1": 2 * prec * rec / max(1e-12, prec + rec),
        "n_pos": len(pos_vals), "n_neg": len(neg_vals),
    }


# ==================================================================
# 메인
# ==================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Step3: solo 형광 절댓값 기반 음성 기준선 산출")
    ap.add_argument("--weights", default=str(P.WEIGHTS_PATH))
    ap.add_argument("--solo_train_root", default=str(P.SOLO_TRAIN))
    ap.add_argument("--solo_test_root", default=str(P.SOLO_TEST))
    ap.add_argument("--out_dir", default=str(P.OUT_STEP3))
    ap.add_argument("--iou", type=float, default=P.IOU)
    ap.add_argument("--imgsz", type=int, default=P.IMG_SIZE)
    ap.add_argument("--device", default="")
    ap.add_argument("--class_tube_name", default="tube")
    ap.add_argument("--class_roi_name", default="roi")
    ap.add_argument("--save_viz", action="store_true",
                    help="검출 오버레이 이미지 저장 (기본 꺼짐, 용량 주의)")
    args = ap.parse_args()

    # conf 는 step1 에서 확정된 값으로 고정한다
    CONF = P.CONF_MIN
    METHOD, METRIC = "G", "p95"

    train_root = Path(args.solo_train_root)
    test_root = Path(args.solo_test_root)
    P.check(train_root, test_root, Path(args.weights))

    out_dir = P.ensure_dir(Path(args.out_dir))
    viz_dir = P.ensure_dir(out_dir / "viz") if args.save_viz else None

    print("=" * 60)
    print("Step3 · 음성 기준선 산출")
    print("=" * 60)
    print(f"  데이터   : {train_root.parent}")
    print(f"  가중치   : {Path(args.weights).name}")
    print(f"  설정     : conf={CONF}, method={METHOD}, metric={METRIC}")
    print(f"  출력     : {out_dir}")
    print(f"  오버레이 : {'저장' if args.save_viz else '생략'}")
    print()

    # ---- 모델 ----
    model = YOLO(str(args.weights))
    names = model.model.names if hasattr(model.model, "names") else model.names
    try:
        tube_cls = next(k for k, v in names.items()
                        if str(v).lower() == args.class_tube_name.lower())
        roi_cls = next(k for k, v in names.items()
                       if str(v).lower() == args.class_roi_name.lower())
    except StopIteration:
        raise SystemExit(f"클래스 tube/roi 를 찾을 수 없습니다. names={names}")

    # ---- 추론 ----
    imgs = list_images(train_root) + list_images(test_root)
    if not imgs:
        raise SystemExit("solo 이미지를 찾지 못했습니다.")
    print(f"[INFO] 대상 이미지 {len(imgs)}장\n")

    rows, skipped = [], []
    train_resolved = train_root.resolve()

    for i, ip in enumerate(imgs, 1):
        label = solo_label_from_path(ip)
        if label is None:
            skipped.append(f"LABEL_UNKNOWN\t{ip}")
            continue
        split = "train" if train_resolved in ip.resolve().parents else "test"

        img = imread_unicode(ip)
        if img is None:
            skipped.append(f"IMREAD_FAIL\t{ip}")
            continue

        res = model.predict(source=img, imgsz=args.imgsz, conf=CONF,
                            iou=args.iou, device=args.device, verbose=False)[0]

        roi_boxes, tube_boxes = [], []
        for b, c, cf in zip(res.boxes.xyxy.cpu().numpy(),
                            res.boxes.cls.cpu().numpy(),
                            res.boxes.conf.cpu().numpy()):
            if int(c) == roi_cls:
                roi_boxes.append((b, float(cf)))
            elif int(c) == tube_cls:
                tube_boxes.append((b, float(cf)))

        if not roi_boxes:
            skipped.append(f"NO_ROI\t{ip}")
            continue

        vals = [g_p95_intensity(safe_crop(img, b)) for b, _ in roi_boxes]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            skipped.append(f"NO_VALID_ROI\t{ip}")
            continue

        # ROI 가 여러 개면 가장 밝은 것을 이미지 대표값으로 삼는다
        rows.append({
            "image_id": ip.stem,
            "image_path": str(ip),
            "split": split,
            "label": label,
            "n_roi": len(roi_boxes),
            "I": f"{float(np.max(vals)):.6f}",
        })

        if viz_dir is not None:
            draw = img.copy()
            for b, cf in tube_boxes:
                draw_box_with_label(draw, b, (0, 255, 0), f"tube {cf:.2f}")
            for b, cf in roi_boxes:
                draw_box_with_label(draw, b, (255, 0, 255), f"roi {cf:.2f}")
            save_viz(draw, viz_dir / ip.stem)

        if i % 20 == 0 or i == len(imgs):
            print(f"  [{i}/{len(imgs)}] {ip.name}")

    if skipped:
        (out_dir / "skipped_images.txt").write_text(
            "\n".join(skipped), encoding="utf-8")
        print(f"\n[WARN] 제외된 이미지 {len(skipped)}장 → skipped_images.txt")

    # ---- 값 저장 ----
    values_csv = out_dir / "solo_values.csv"
    with open(values_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "image_id", "image_path", "split", "label", "n_roi", "I"])
        w.writeheader()
        w.writerows(rows)

    def pick(split, label):
        return [float(r["I"]) for r in rows
                if r["split"] == split and r["label"] == label]

    train_neg = pick("train", "neg")
    train_pos = pick("train", "pos")
    test_neg = pick("test", "neg")
    test_pos = pick("test", "pos")

    print(f"\n[데이터] train neg={len(train_neg)} pos={len(train_pos)} / "
          f"test neg={len(test_neg)} pos={len(test_pos)}")

    if len(train_neg) < 3:
        raise SystemExit("train 음성 샘플이 3개 미만이라 기준선을 산출할 수 없습니다.")

    # ---- 음성 기준선 ----
    sh_W, sh_p = stats.shapiro(train_neg)
    mean_v = float(np.mean(train_neg))
    std_v = float(np.std(train_neg, ddof=1))
    p997 = float(np.percentile(train_neg, 99.7))

    is_normal = bool(sh_p > 0.05)
    cutoff = (mean_v + 3.0 * std_v) if is_normal else p997

    qq_plot(train_neg, out_dir / "neg_baseline_qq.png", "Q-Q plot (train NEG)")

    neg_stats = {
        "n_train_neg": len(train_neg),
        "shapiro_W": float(sh_W),
        "shapiro_p": float(sh_p),
        "is_normal_by_p_gt_0.05": is_normal,
        "mean": mean_v,
        "std": std_v,
        "median": float(np.median(train_neg)),
        "min": float(np.min(train_neg)),
        "max": float(np.max(train_neg)),
        "skew": float(stats.skew(train_neg, bias=False)),
        "kurtosis": float(stats.kurtosis(train_neg, bias=False)),
        "p99_7": p997,
        "mean_plus_3sd": mean_v + 3.0 * std_v,
        "cutoff": cutoff,
        "cutoff_rule": "mean+3SD" if is_normal else "p99.7",
        "fixed_settings": {"conf": CONF, "method": METHOD, "metric": METRIC},
    }
    (out_dir / "neg_baseline_stats.json").write_text(
        json.dumps(neg_stats, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(out_dir / "neg_train_values.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["I_neg_train"])
        w.writerows([[f"{v:.6f}"] for v in train_neg])

    # ---- 평가 ----
    test_eval = eval_cutoff(test_pos, test_neg, cutoff)

    summary = {
        "settings_fixed": {"conf": CONF, "method": METHOD, "metric": METRIC},
        "negative_baseline": neg_stats,
        "test_eval_negative_cut": test_eval,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 출력 ----
    print()
    print("=" * 60)
    print(f"  Shapiro-Wilk : W={sh_W:.4f}, p={sh_p:.4f} "
          f"→ {'정규분포로 볼 수 있음' if is_normal else '정규분포로 보기 어려움'}")
    print(f"  train NEG    : mean={mean_v:.2f}, SD={std_v:.2f}, "
          f"range=[{np.min(train_neg):.1f}, {np.max(train_neg):.1f}]")
    print(f"  mean+3SD     : {mean_v + 3*std_v:.3f}")
    print(f"  p99.7        : {p997:.3f}")
    print(f"  → cutoff     : {cutoff:.3f}  ({neg_stats['cutoff_rule']})")
    print("-" * 60)
    print(f"  Test  ACC={test_eval['ACC']*100:.1f}%  "
          f"TP={test_eval['TP']} FN={test_eval['FN']} "
          f"FP={test_eval['FP']} TN={test_eval['TN']}")
    print("=" * 60)
    print(f"\n[저장] {out_dir}")


if __name__ == "__main__":
    main()