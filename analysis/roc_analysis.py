# -*- coding: utf-8 -*-
"""
roc_analysis.py

목적
    임계값에 의존하지 않는 성능 지표를 낸다.

    step3 와 step4 에서 구한 정확도는 특정 임계값(221.0, 1.1162)을 전제한
    값이다. 임계값을 바꾸면 숫자도 바뀌므로, 그 값만으로는 "신호 자체가
    얼마나 좋은가"를 알 수 없다.

    ROC 곡선은 가능한 모든 임계값을 훑으면서 민감도와 위양성률의 관계를
    그린다. 그 아래 면적(AUC)은 임계값 선택과 무관하며,
    "임의의 양성 표본이 임의의 음성 표본보다 높은 점수를 받을 확률"과 같다.

        AUC = 0.5   무작위와 다름없음
        AUC = 1.0   완전한 분리

두 가지를 따로 본다
    solo — G_p95 절댓값으로 양성/음성을 가른다.
           step3 의 cutoff 가 다루는 문제다.

    pair — ratio = I_sample / I_nc 로 가른다.
           step4 의 T_ratio 가 다루는 문제이며, 실제 앱이 쓰는 방식이다.
           neg_pos 를 양성, neg_neg 를 음성으로 놓는다.

train / test 분리
    원본은 solo train 과 test 를 합쳐서 계산했다. train 은 임계값을 정하는
    데 쓴 데이터이므로, 합치면 평가가 낙관적이 된다.
    여기서는 train, test, 전체를 각각 계산한다.

신뢰구간
    AUC 에도 부트스트랩 신뢰구간을 붙인다. 표본이 적으면 AUC 가 1.0 에
    가깝게 나와도 구간이 넓을 수 있다.

출력
    results/roc/
      ├── solo_roc_{split}.csv / .png
      ├── pair_roc.csv / .png
      ├── roc_values.csv          이미지별 점수와 라벨
      └── summary.json

실행
    python analysis/roc_analysis.py

원본 대비 수정 사항
    - solo 의 train / test 를 분리해 각각 계산
    - AUC 부트스트랩 신뢰구간 추가
    - 임계값별 민감도·특이도 표 출력 (현재 운영값이 곡선의 어디인지)
    - 하드코딩 경로 제거, 오버레이 저장 기본 끄기, 한글 경로 대응

실행 결과 (2026-08, weights.pt)

    solo · G_p95 절댓값
        split    n    pos  neg     AUC        95% CI
        train    80    40   40   0.9994  [0.9972, 1.0000]
        test     30    15   15   1.0000  [1.0000, 1.0000]
        전체    110    55   55   0.9990  [0.9960, 1.0000]

    pair · ratio = I_sample / I_nc
        n=44 (neg_pos 20, neg_neg 24)   AUC = 0.9646  [0.9027, 0.9979]

    절댓값으로는 거의 완전히 갈리지만(AUC 0.999), 비율로 바꾸면 성능이
    떨어진다(0.965). 비율은 기기 차이를 상쇄해 주는 대신 NC 튜브의 변동을
    새로 끌어들이기 때문이다. 분모가 흔들리면 결과도 흔들린다.

    이는 비율 방식이 나쁘다는 뜻이 아니다. 기기별 NC 형광값이 최대 19%
    차이 나므로(step4 참고) 절댓값 판정은 기기를 바꾸면 무너진다.
    비율은 그 문제를 해결하는 대신 다른 대가를 치르는 선택이며,
    그 대가가 AUC 0.999 → 0.965 로 정량화된다.

    solo test 의 AUC 1.0000 은 액면 그대로 받아들이면 안 된다. test 양성
    15장 중 5장이 G_p95 = 255 로 포화되어 점수가 위로 몰린 결과다(step3 참고).
    train 의 0.9994 가 더 현실적인 값이다.

    pair 임계값별 민감도 / 특이도
        T=1.0500   민감도 100.0%   특이도  83.3%
        T=1.1162   민감도 100.0%   특이도  83.3%   ← 현재
        T=1.1480   민감도  90.0%   특이도  83.3%
        T=1.1800   민감도  85.0%   특이도  87.5%
        T=1.2000   민감도  85.0%   특이도  95.8%

    현재 값은 민감도를 100% 로 유지하는 구간의 위쪽 끝에 있다.
    여기서 더 올리면 민감도가 먼저 떨어지고 특이도는 늦게 오른다.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as P

try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit("ultralytics 가 필요합니다:  pip install ultralytics") from e


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
N_BOOTSTRAP = 2000


def list_images(root):
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTS)


def imread_unicode(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def safe_crop(img, xyxy):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    H, W = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def g_p95(crop):
    if crop is None:
        return np.nan
    g = crop[:, :, 1].astype(np.float32)
    return float(np.percentile(g, 95.0)) if g.size else np.nan


def center_y(b):
    return (float(b[1]) + float(b[3])) / 2.0


def solo_label(path):
    low = str(path).lower().replace("\\", "/")
    if "/pos/" in low:
        return "pos"
    if "/neg/" in low:
        return "neg"
    return None


def auc_of(scores, labels):
    if len(np.unique(labels)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(auc(fpr, tpr))


def bootstrap_auc_ci(scores, labels, n=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    if len(scores) < 6:
        return None
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(scores), len(scores))
        s, l = scores[idx], labels[idx]
        if len(np.unique(l)) < 2:
            continue
        out.append(auc_of(s, l))
    if len(out) < n * 0.5:
        return None
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi), "n_boot": len(out)}


def save_roc(out_prefix: Path, scores, labels, title: str, mark=None):
    """
    mark: (임계값, 라벨) — 현재 운영값이 곡선의 어디인지 표시한다.
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    if len(scores) == 0 or len(np.unique(labels)) < 2:
        return None

    fpr, tpr, thr = roc_curve(labels, scores)
    a = float(auc(fpr, tpr))
    ci = bootstrap_auc_ci(scores, labels)

    with open(out_prefix.with_suffix(".csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["fpr", "tpr", "threshold"])
        w.writerows(zip(fpr, tpr, thr))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    lbl = f"AUC = {a:.3f}"
    if ci:
        lbl += f"\n95% CI [{ci['lo']:.3f}, {ci['hi']:.3f}]"
    ax.plot(fpr, tpr, linewidth=2, label=lbl)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    if mark is not None:
        T, name = mark
        i = int(np.argmin(np.abs(thr - T)))
        ax.plot(fpr[i], tpr[i], "o", markersize=9, color="crimson",
                label=f"{name} (T={T:g})")

    ax.set_xlabel("False Positive Rate  (1 - Specificity)")
    ax.set_ylabel("True Positive Rate  (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".png"), dpi=200)
    plt.close(fig)

    return {"auc": a, "ci": ci, "n": len(scores),
            "n_pos": int((labels == 1).sum()), "n_neg": int((labels == 0).sum()),
            "fpr": fpr, "tpr": tpr, "thr": thr}


def print_operating_point(res, T, name):
    """현재 운영 임계값이 곡선의 어느 지점인지 보여준다."""
    if res is None:
        return
    thr = res["thr"]
    i = int(np.argmin(np.abs(thr - T)))
    print(f"    {name} T={T:g} 에서  민감도 {res['tpr'][i]*100:5.1f}%  "
          f"특이도 {(1-res['fpr'][i])*100:5.1f}%")


def main():
    ap = argparse.ArgumentParser(description="ROC / AUC 분석")
    ap.add_argument("--weights", default=str(P.WEIGHTS_PATH))
    ap.add_argument("--solo_train_root", default=str(P.SOLO_TRAIN))
    ap.add_argument("--solo_test_root", default=str(P.SOLO_TEST))
    ap.add_argument("--pair_negpos_root", default=str(P.PAIR_NEGPOS))
    ap.add_argument("--pair_negneg_root", default=str(P.PAIR_NEGNEG))
    ap.add_argument("--out_dir", default=str(P.OUT_ROC))
    ap.add_argument("--iou", type=float, default=P.IOU)
    ap.add_argument("--imgsz", type=int, default=P.IMG_SIZE)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    CONF = P.CONF_MIN
    out_dir = P.ensure_dir(Path(args.out_dir))
    P.check(Path(args.weights), Path(args.solo_train_root), Path(args.pair_negpos_root))

    print("=" * 66)
    print("ROC / AUC · 임계값에 의존하지 않는 성능 지표")
    print("=" * 66)
    print(f"  가중치 : {Path(args.weights).name}")
    print(f"  설정   : conf={CONF}, G채널 p95")
    print(f"  출력   : {out_dir}")
    print()

    model = YOLO(str(args.weights))
    names = model.model.names if hasattr(model.model, "names") else model.names
    roi_id = next(k for k, v in names.items() if str(v).lower() == "roi")

    def detect_rois(path: Path):
        img = imread_unicode(path)
        if img is None:
            return None, []
        r = model.predict(source=img, imgsz=args.imgsz, conf=CONF,
                          iou=args.iou, device=args.device, verbose=False)[0]
        rois = [b for b, c in zip(r.boxes.xyxy.cpu().numpy(),
                                  r.boxes.cls.cpu().numpy().astype(int))
                if c == roi_id]
        return img, rois

    rows = []

    # ================= SOLO =================
    print("[1/2] solo · G_p95 절댓값")
    train_resolved = Path(args.solo_train_root).resolve()
    solo_imgs = list_images(args.solo_train_root) + list_images(args.solo_test_root)

    for i, ip in enumerate(solo_imgs, 1):
        lab = solo_label(ip)
        if lab is None:
            continue
        img, rois = detect_rois(ip)
        if img is None or not rois:
            continue
        vals = [g_p95(safe_crop(img, b)) for b in rois]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            continue
        rows.append({
            "dataset": "solo",
            "split": "train" if train_resolved in ip.resolve().parents else "test",
            "image_id": ip.stem, "label": lab, "y": 1 if lab == "pos" else 0,
            "score": f"{float(np.max(vals)):.6f}",
        })
        if i % 30 == 0 or i == len(solo_imgs):
            print(f"    {i}/{len(solo_imgs)}")

    # ================= PAIR =================
    print("\n[2/2] pair · ratio = I_sample / I_nc")

    def scan_pair(root, group, y):
        imgs = list_images(root)
        print(f"    {group}: {len(imgs)}장")
        for ip in imgs:
            img, rois = detect_rois(ip)
            if img is None or len(rois) < 2:
                continue
            rs = sorted(rois, key=center_y)
            Iu, Il = g_p95(safe_crop(img, rs[0])), g_p95(safe_crop(img, rs[1]))
            if not (np.isfinite(Iu) and np.isfinite(Il) and Iu > 0):
                continue
            rows.append({"dataset": "pair", "split": group,
                         "image_id": ip.stem,
                         "label": "pos" if y else "neg", "y": y,
                         "score": f"{Il / Iu:.6f}"})

    scan_pair(args.pair_negpos_root, "neg_pos", 1)
    if Path(args.pair_negneg_root).exists():
        scan_pair(args.pair_negneg_root, "neg_neg", 0)

    with open(out_dir / "roc_values.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ================= 계산 =================
    def pick(dataset, split=None):
        sub = [r for r in rows if r["dataset"] == dataset
               and (split is None or r["split"] == split)]
        return (np.array([float(r["score"]) for r in sub]),
                np.array([int(r["y"]) for r in sub]))

    summary = {}

    print()
    print("=" * 66)
    print("solo · G_p95 절댓값 기준")
    print("=" * 66)
    for split, name in ((None, "all"), ("train", "train"), ("test", "test")):
        s, y = pick("solo", split)
        if len(s) == 0 or len(np.unique(y)) < 2:
            continue
        res = save_roc(out_dir / f"solo_roc_{name}", s, y,
                       f"ROC · solo {name} (G_p95)",
                       mark=(P.ABS_NEG_CUTOFF, "step3 cutoff"))
        ci = res["ci"]
        print(f"  [{name:5s}] n={res['n']:3d} (pos {res['n_pos']}, neg {res['n_neg']})  "
              f"AUC = {res['auc']:.4f}"
              + (f"   95% CI [{ci['lo']:.4f}, {ci['hi']:.4f}]" if ci else ""))
        print_operating_point(res, P.ABS_NEG_CUTOFF, "step3 cutoff")
        summary[f"solo_{name}"] = {"auc": res["auc"], "ci": ci,
                                   "n_pos": res["n_pos"], "n_neg": res["n_neg"]}

    print()
    print("=" * 66)
    print("pair · ratio 기준  (앱이 실제로 쓰는 방식)")
    print("=" * 66)
    s, y = pick("pair")
    if len(s) and len(np.unique(y)) == 2:
        res = save_roc(out_dir / "pair_roc", s, y,
                       "ROC · pair (ratio = sample / NC)",
                       mark=(P.RATIO_THR, "T_ratio"))
        ci = res["ci"]
        print(f"  n={res['n']} (neg_pos {res['n_pos']}, neg_neg {res['n_neg']})  "
              f"AUC = {res['auc']:.4f}"
              + (f"   95% CI [{ci['lo']:.4f}, {ci['hi']:.4f}]" if ci else ""))
        print_operating_point(res, P.RATIO_THR, "T_ratio")
        summary["pair"] = {"auc": res["auc"], "ci": ci,
                           "n_pos": res["n_pos"], "n_neg": res["n_neg"]}

        # 곡선 위 몇 지점을 표로
        print()
        print("    임계값별 민감도 / 특이도")
        thr, tpr, fpr = res["thr"], res["tpr"], res["fpr"]
        seen = set()
        for t in (1.05, 1.08, 1.10, 1.1162, 1.148, 1.18, 1.20):
            i = int(np.argmin(np.abs(thr - t)))
            if i in seen:
                continue
            seen.add(i)
            tag = ""
            if abs(t - P.RATIO_THR) < 1e-6:
                tag = "  ← 현재"
            print(f"      T={t:<7.4f}  민감도 {tpr[i]*100:5.1f}%   "
                  f"특이도 {(1-fpr[i])*100:5.1f}%{tag}")
    else:
        print("  음성 pair 가 없어 계산 불가")

    (out_dir / "summary.json").write_text(
        json.dumps({"settings": {"conf": CONF, "method": "G", "metric": "p95"},
                    "results": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    print()
    print("=" * 66)
    print("  AUC 는 임계값 선택과 무관한 지표다.")
    print("  임계값에 의존하는 정확도보다 신호 자체의 품질을 보여준다.")
    print("=" * 66)
    print(f"\n[저장] {out_dir}")


if __name__ == "__main__":
    main()
