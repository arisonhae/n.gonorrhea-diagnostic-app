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
    3. 음성 기준선을 세 방식으로 산출해 비교한다.

       [A]  99.7 백분위수 — 음성 분포만 사용. 원본 방식.
            표본이 적으면 이 값은 최댓값과 같아지므로,
            가장 밝은 음성 이미지 한 장이 기준선을 결정하게 된다.

       [A'] mean + 3SD — 정규성을 전제한다.
            Shapiro-Wilk 로 가정이 성립하는지 함께 확인한다.

       [B]  Youden's J — 양성과 음성을 모두 사용해
            민감도+특이도가 최대가 되는 지점을 찾는다.
            음성만 보는 방식과 달리 양성 정보를 버리지 않는다.

       세 값 모두 부트스트랩 신뢰구간을 산출한다.

    4. test 데이터에 적용해 혼동행렬과 Wilson 신뢰구간을 낸다.
       세 방식을 각각 적용했을 때 결과가 어떻게 달라지는지도 함께 보인다.

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

실행
    python analysis/step3_threshold_analysis.py

    # 다른 기준선 방식으로 계산
    python analysis/step3_threshold_analysis.py --cutoff_rule youden

수정 이력
    - 기준선 산출을 세 방식(99.7 백분위수 / mean+3SD / Youden's J) 비교로
      확장하고, 각각에 부트스트랩 신뢰구간을 추가
    - 정확도·민감도·특이도에 Wilson 신뢰구간 추가
    - 하드코딩 경로를 paths.py 로 이관
    - 오버레이 이미지 저장을 기본 끄기로 변경 (PNG 누적 용량 문제)
    - 저장 시에도 JPEG + 리사이즈로 용량 축소

실행 결과 (2026-08, weights.pt, solo 110장)

    cutoff = 221.0 (99.7 백분위수) 유지

        방식              T        95% CI           test 결과
        p99.7         221.000  [213.88, 221.00]   TP15 FN0 FP1 TN14  96.7%
        mean+3SD      240.627   —                 TP11 FN4 FP0 TN15  86.7%
        Youden's J    221.500  [217.00, 223.00]   TP15 FN0 FP1 TN14  96.7%

    n=40 에서 99.7 백분위수는 최댓값과 같아지므로, 가장 밝은 음성 이미지
    한 장이 기준선을 결정하는 구조인 것은 사실이다. 그러나 양성 정보까지
    활용하는 Youden's J 로 재산출한 값이 221.5 로 거의 같았고 test 성능도
    동일했다. 방법을 바꿀 근거가 없다.

    정규성을 전제한 mean+3SD 는 위음성 4건을 발생시켜 정확도가 86.7% 로
    떨어진다. Shapiro-Wilk p<0.0001 로 가정이 기각되므로 부적절하며,
    포스터의 µ+3σ 표기는 오류다.

    부트스트랩 신뢰구간의 상한이 221.00 인 것은 최댓값에 막혀 있기 때문이다.
    리샘플링해도 최댓값을 넘을 수 없다. Youden 기준 [217.00, 223.00] 쪽이
    불확실성을 더 정직하게 표현한다.

    test 성능 (95% CI)
        정확도  96.7%  29/30   83.3 – 99.4%
        민감도 100.0%  15/15   79.6 – 100.0%
        특이도  93.3%  14/15   70.2 –  98.8%

    위양성 1건은 solo_test_neg_tube_1_2 (I=222.0) 로 cutoff 와 1 차이다.
    train 음성의 최댓값이 221 이어서 생긴 일이며, 음성 표본을 늘리면
    기준선이 올라가 이 오판이 사라질 가능성이 높다.
    방법론의 문제가 아니라 표본 수의 문제다.

    test set 15장 중 5장(blue, 2x, r_0)은 촬영 조건 최적화 실험에서 나온
    이미지이며, 양성 5장이 모두 G_p95 = 255 로 포화되었다. 포화된 값은
    무조건 cutoff 를 넘으므로 사실상 자동 정답이 된다. 기본 조건(tube_*)
    20장만으로 계산하면 정확도는 95.0% 다.
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


def youden_threshold(pos_vals, neg_vals):
    """
    민감도 + 특이도 - 1 이 최대가 되는 지점.
    음성 분포만 보는 백분위수 방식과 달리 양성 정보까지 사용한다.
    """
    v = np.concatenate([np.asarray(pos_vals, float), np.asarray(neg_vals, float)])
    y = np.concatenate([np.ones(len(pos_vals), int), np.zeros(len(neg_vals), int)])
    m = np.isfinite(v)
    v, y = v[m], y[m]
    if v.size < 4 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan, np.nan

    order = np.argsort(v)
    v, y = v[order], y[order]
    P_, N_ = int((y == 1).sum()), int((y == 0).sum())
    TP_right = np.cumsum((y == 1).astype(int)[::-1])[::-1]
    FP_right = np.cumsum((y == 0).astype(int)[::-1])[::-1]

    best = (-1.0, np.nan, np.nan, np.nan)
    for i in np.where(np.diff(v) != 0)[0]:
        TPR = TP_right[i + 1] / P_
        FPR = FP_right[i + 1] / N_
        J = TPR - FPR
        if J > best[0]:
            best = (J, (v[i] + v[i + 1]) / 2.0, TPR, FPR)
    J, T, TPR, FPR = best
    return float(T), float(J), float(TPR), float(FPR)


def bootstrap_ci(func, *arrays, n=2000, seed=0, alpha=0.05):
    """같은 길이의 배열들을 함께 리샘플링해 func 결과의 신뢰구간을 구한다."""
    rng = np.random.default_rng(seed)
    size = len(arrays[0])
    if size < 3:
        return None
    out = []
    for _ in range(n):
        idx = rng.integers(0, size, size)
        try:
            val = func(*[np.asarray(a)[idx] for a in arrays])
        except Exception:
            continue
        if val is not None and np.isfinite(val):
            out.append(val)
    if len(out) < n * 0.5:
        return None
    lo, hi = np.percentile(out, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return {"lo": float(lo), "hi": float(hi), "n_boot": len(out)}


def wilson_ci(k, n, z=1.96):
    """이항 비율의 Wilson 신뢰구간."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    s = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return {"point": p, "lo": float((c - s) / d), "hi": float((c + s) / d),
            "k": int(k), "n": int(n)}


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
    ap.add_argument("--cutoff_rule", choices=["p99.7", "mean3sd", "youden"],
                    default="p99.7",
                    help="음성 기준선 산출 방식. 기본은 원본과 같은 p99.7")
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

    # ---- 음성 기준선: 세 방식 비교 ----
    sh_W, sh_p = stats.shapiro(train_neg)
    mean_v = float(np.mean(train_neg))
    std_v = float(np.std(train_neg, ddof=1))
    p997 = float(np.percentile(train_neg, 99.7))
    p95 = float(np.percentile(train_neg, 95.0))
    max_v = float(np.max(train_neg))

    is_normal = bool(sh_p > 0.05)

    print()
    print("=" * 64)
    print("음성 기준선 산출 · 세 방식 비교")
    print("=" * 64)
    print(f"  train 음성 n={len(train_neg)}  "
          f"mean={mean_v:.2f}  SD={std_v:.2f}  range=[{np.min(train_neg):.1f}, {max_v:.1f}]")
    print(f"  Shapiro-Wilk  W={sh_W:.4f}  p={sh_p:.4f}  "
          f"→ {'정규분포로 볼 수 있음' if is_normal else '정규분포로 보기 어려움'}")
    print()

    # [A] 백분위수 (원본 방식)
    T_a = p997
    ci_a = bootstrap_ci(lambda a: np.percentile(a, 99.7), train_neg)
    print(f"  [A] 99.7 백분위수 · 음성 분포만 사용  (원본 방식)")
    print(f"      T = {T_a:.3f}")
    if ci_a:
        print(f"      95% CI = [{ci_a['lo']:.2f}, {ci_a['hi']:.2f}]")
    if abs(T_a - max_v) < 1e-6:
        print(f"      ※ n={len(train_neg)} 에서 99.7 백분위수는 최댓값과 같아진다.")
        print(f"         단일 극단값이 기준선 전체를 결정한다.")

    # [A'] 참고: mean + 3SD
    T_a2 = mean_v + 3.0 * std_v
    print(f"  [A'] mean + 3SD  = {T_a2:.3f}   "
          f"({'정규성이 성립하지 않아 근거가 약하다' if not is_normal else '참고값'})")

    # [B] Youden's J — 양성 정보까지 사용
    T_b, J, TPR, FPR = youden_threshold(train_pos, train_neg)
    ci_b = bootstrap_ci(
        lambda p_, n_: youden_threshold(p_, n_)[0],
        train_pos, train_neg) if len(train_pos) == len(train_neg) else None
    print(f"\n  [B] Youden's J · 양성과 음성을 모두 사용")
    if np.isfinite(T_b):
        print(f"      T = {T_b:.3f}  (J={J:.3f}, TPR={TPR:.3f}, FPR={FPR:.3f})")
        if ci_b:
            print(f"      95% CI = [{ci_b['lo']:.2f}, {ci_b['hi']:.2f}]")
    else:
        print("      계산 불가 (표본 부족)")

    # 사용할 값 선택
    T_map = {"p99.7": T_a, "mean3sd": T_a2, "youden": T_b}
    cutoff = T_map[args.cutoff_rule]
    print(f"\n  → 사용할 기준선: {cutoff:.3f}  ({args.cutoff_rule})\n")

    qq_plot(train_neg, out_dir / "neg_baseline_qq.png", "Q-Q plot (train NEG)")

    neg_stats = {
        "n_train_neg": len(train_neg),
        "n_train_pos": len(train_pos),
        "shapiro_W": float(sh_W),
        "shapiro_p": float(sh_p),
        "is_normal_by_p_gt_0.05": is_normal,
        "mean": mean_v,
        "std": std_v,
        "median": float(np.median(train_neg)),
        "min": float(np.min(train_neg)),
        "max": max_v,
        "skew": float(stats.skew(train_neg, bias=False)),
        "kurtosis": float(stats.kurtosis(train_neg, bias=False)),
        "method_A_percentile": {
            "description": "음성 분포의 99.7 백분위수. 원본 방식.",
            "p99_7": p997, "p95": p95,
            "equals_max": bool(abs(p997 - max_v) < 1e-6),
            "bootstrap_ci_95": ci_a,
        },
        "method_A2_mean_plus_3sd": {
            "description": "평균 + 3표준편차. 정규성을 전제한다.",
            "value": T_a2,
            "assumption_valid": is_normal,
        },
        "method_B_youden": {
            "description": "민감도+특이도 최대 지점. 양성 정보까지 사용한다.",
            "T": T_b if np.isfinite(T_b) else None,
            "youden_J": J if np.isfinite(J) else None,
            "TPR": TPR if np.isfinite(TPR) else None,
            "FPR": FPR if np.isfinite(FPR) else None,
            "bootstrap_ci_95": ci_b,
        },
        "cutoff_rule": args.cutoff_rule,
        "cutoff": cutoff,
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
        "test_eval_ci": {
            "accuracy": wilson_ci(test_eval["TP"] + test_eval["TN"],
                                  test_eval["n_pos"] + test_eval["n_neg"]),
            "sensitivity": wilson_ci(test_eval["TP"], test_eval["n_pos"]),
            "specificity": wilson_ci(test_eval["TN"], test_eval["n_neg"]),
        },
        "test_eval_by_rule": {
            name: eval_cutoff(test_pos, test_neg, T)
            for name, T in (("p99.7", T_a), ("mean3sd", T_a2), ("youden", T_b))
            if np.isfinite(T)
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 출력 ----
    print("=" * 64)
    print("Test 평가")
    print("=" * 64)
    acc_ci = wilson_ci(test_eval["TP"] + test_eval["TN"],
                       test_eval["n_pos"] + test_eval["n_neg"])
    sens_ci = wilson_ci(test_eval["TP"], test_eval["n_pos"])
    spec_ci = wilson_ci(test_eval["TN"], test_eval["n_neg"])
    print(f"  TP={test_eval['TP']}  FN={test_eval['FN']}  "
          f"FP={test_eval['FP']}  TN={test_eval['TN']}")
    if acc_ci:
        print(f"  정확도  {acc_ci['point']*100:5.1f}%  ({acc_ci['k']}/{acc_ci['n']})"
              f"   95% CI {acc_ci['lo']*100:.1f}–{acc_ci['hi']*100:.1f}%")
    if sens_ci:
        print(f"  민감도  {sens_ci['point']*100:5.1f}%  ({sens_ci['k']}/{sens_ci['n']})"
              f"   95% CI {sens_ci['lo']*100:.1f}–{sens_ci['hi']*100:.1f}%")
    if spec_ci:
        print(f"  특이도  {spec_ci['point']*100:5.1f}%  ({spec_ci['k']}/{spec_ci['n']})"
              f"   95% CI {spec_ci['lo']*100:.1f}–{spec_ci['hi']*100:.1f}%")

    # 다른 방식을 썼다면 결과가 어떻게 달라지는지
    print()
    print("-" * 64)
    print("  기준선을 바꾸면 test 결과가 어떻게 달라지는가")
    print("-" * 64)
    for name, T in (("p99.7  ", T_a), ("mean+3SD", T_a2), ("youden ", T_b)):
        if not np.isfinite(T):
            continue
        e = eval_cutoff(test_pos, test_neg, T)
        mark = " ←" if abs(T - cutoff) < 1e-9 else ""
        print(f"  {name} T={T:7.2f}   TP={e['TP']:2d} FN={e['FN']:2d} "
              f"FP={e['FP']:2d} TN={e['TN']:2d}   ACC={e['ACC']*100:5.1f}%{mark}")

    print("=" * 64)
    print(f"\n[저장] {out_dir}")


if __name__ == "__main__":
    main()