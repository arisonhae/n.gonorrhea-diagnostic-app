# -*- coding: utf-8 -*-
"""
step2b_signal_separation.py

목적
    step2a 에서 고른 G · p95 로 양성과 음성이 실제로 분리되는지 검정한다.

    step2a 가 "무엇으로 측정할까"를 정했다면, 여기서는 "그 측정값으로
    두 집단이 정말 갈라지는가"를 확인한다.

검정 방법에 대하여
    원본은 Welch t-test 만 사용했다. 그런데 이 검정은 각 집단이 정규분포를
    따른다고 가정한다. step3 에서 train 음성 분포에 Shapiro-Wilk 를 적용한
    결과 p < 0.0001 로 정규성이 기각되었으므로, 가정이 성립하지 않을 수 있다.

    그래서 여기서는 세 가지를 함께 낸다.

        Shapiro-Wilk   각 집단이 정규분포를 따르는가
        Welch t-test   평균 차이 (정규성 가정)
        Mann-Whitney U 순위 기반 (정규성 가정 없음)

    두 검정의 결론이 같으면 가정 위반이 결과를 바꾸지 않았다는 뜻이므로
    안심하고 t-test 결과를 쓸 수 있다. 다르면 비모수 결과를 따라야 한다.

    효과크기도 두 가지를 낸다.

        Cohen's d      평균 차이를 표준편차로 나눈 값 (정규성 전제)
        Cliff's delta  임의의 양성 값이 임의의 음성 값보다 클 확률에서
                       그 반대 확률을 뺀 값. -1 ~ 1 범위이며 분포 모양에
                       영향을 받지 않는다.

train / test 분리
    전체를 합쳐서 보면 학습에 쓴 데이터가 섞여 결과가 낙관적이 된다.
    train, test, 전체를 각각 계산해 비교한다.

출력
    results/step2b_separation/
      ├── Gp95_values.csv     이미지별 G_p95
      ├── Gp95_stats.csv      split 별 검정 결과
      └── summary.json

실행
    python analysis/step2b_signal_separation.py

원본 대비 수정 사항
    - 정규성 검정과 비모수 검정 추가 (Welch t-test 의 가정 확인)
    - Cliff's delta 추가
    - train / test 분리 집계
    - 하드코딩 경로 제거, 한글 경로 대응

실행 결과 (2026-08, weights.pt, solo 110장)

    G_p95 로 양성과 음성이 명확히 분리된다.

                양성        음성      Welch p     M-W p      d      delta
    train    235.6±9.1  204.6±12.0   1.1e-20   1.5e-14   2.916   0.999
    test     246.1±9.3  205.1± 7.7   2.6e-13   3.1e-06   4.809   1.000
    전체     238.5±10.2 204.7±11.0   1.1e-31   1.8e-19   3.191   0.998

    정규성은 대부분의 조합에서 성립하지 않았다 (Shapiro-Wilk p < 0.05).
    Welch t-test 는 정규성을 가정하므로, 순위 기반의 Mann-Whitney U 를
    함께 수행했다. 두 검정의 결론이 일치하므로 가정 위반이 결과를
    뒤집지는 않았다.

    Cliff's delta 는 전체 기준 0.998 로 거의 완전한 분리를 뜻하지만,
    겹치는 구간 [221.0, 222.0] 에 6개 표본이 존재한다. step3 의 cutoff
    221.0 이 바로 이 구간에 놓이며, test 위양성 1건이 여기서 발생했다.
    분리 자체가 나쁜 것이 아니라 임계값이 경계에 걸려 있는 것이다.

    test 의 효과크기(d=4.809)가 train(2.916)보다 큰 것은 성능이 좋아서가
    아니다. test 양성 15장 중 5장이 G_p95 = 255 로 포화되어 값이 위로
    몰리고 분산이 줄어든 결과다. train 의 2.916 이 더 현실적인 값이다.

    train 음성 분포는 W=0.804 로 정규성에서 가장 멀다. 평균 204.55 보다
    중앙값 208.00 이 높아 왼쪽으로 긴 꼬리를 가진 형태다.
"""

import argparse
import csv
import json
import sys
from math import sqrt
from pathlib import Path

import cv2
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as P

try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit("ultralytics 가 필요합니다:  pip install ultralytics") from e


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SATURATION_LEVEL = 254.0


# ==================================================================
# 유틸
# ==================================================================
def list_images(roots):
    paths = []
    for r in roots:
        r = Path(r)
        if r.is_dir():
            paths.extend(p for p in r.rglob("*") if p.suffix.lower() in IMG_EXTS)
        elif r.suffix.lower() in IMG_EXTS:
            paths.append(r)
    return sorted(set(paths))


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
        return np.nan, np.nan
    g = crop[:, :, 1].astype(np.float32)
    if g.size == 0:
        return np.nan, np.nan
    return float(np.percentile(g, 95.0)), float(np.mean(g >= SATURATION_LEVEL))


def solo_label(path):
    low = str(path).lower().replace("\\", "/")
    if "/pos/" in low:
        return "pos"
    if "/neg/" in low:
        return "neg"
    return None


# ==================================================================
# 통계
# ==================================================================
def cohens_d(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return np.nan
    n1, n2 = len(pos), len(neg)
    sp = sqrt(((n1 - 1) * np.var(pos, ddof=1) + (n2 - 1) * np.var(neg, ddof=1))
              / (n1 + n2 - 2))
    return float((np.mean(pos) - np.mean(neg)) / sp) if sp > 0 else np.nan


def cliffs_delta(pos, neg):
    """
    P(양성 > 음성) - P(양성 < 음성).
    분포 모양과 무관하며, 1 이면 두 집단이 완전히 분리된다는 뜻이다.
    통상 |0.147| 작음, |0.33| 중간, |0.474| 이상 큼 으로 본다.
    """
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def overlap_count(pos, neg):
    """두 집단의 값 범위가 겹치는 구간에 든 표본 수."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    lo, hi = np.max(neg), np.min(pos)
    if hi > lo:
        return 0, (float(lo), float(hi))       # 완전 분리
    n = int(np.sum(pos <= lo) + np.sum(neg >= hi))
    return n, (float(hi), float(lo))


def analyze(pos, neg, label: str):
    """한 split 에 대한 전체 검정 결과."""
    res = {"split": label, "n_pos": len(pos), "n_neg": len(neg)}
    if len(pos) < 3 or len(neg) < 3:
        res["note"] = "표본 부족"
        return res

    res.update({
        "pos_mean": float(np.mean(pos)), "pos_sd": float(np.std(pos, ddof=1)),
        "pos_median": float(np.median(pos)),
        "neg_mean": float(np.mean(neg)), "neg_sd": float(np.std(neg, ddof=1)),
        "neg_median": float(np.median(neg)),
    })

    # 정규성
    wp, pp = stats.shapiro(pos)
    wn, pn = stats.shapiro(neg)
    res.update({
        "shapiro_pos_W": float(wp), "shapiro_pos_p": float(pp),
        "shapiro_neg_W": float(wn), "shapiro_neg_p": float(pn),
        "both_normal": bool(pp > 0.05 and pn > 0.05),
    })

    # 모수 / 비모수
    t, p_t = stats.ttest_ind(pos, neg, equal_var=False)
    u, p_u = stats.mannwhitneyu(pos, neg, alternative="two-sided")
    res.update({
        "welch_t": float(t), "welch_p": float(p_t),
        "mannwhitney_U": float(u), "mannwhitney_p": float(p_u),
        "same_conclusion": bool((p_t < 0.05) == (p_u < 0.05)),
    })

    # 효과크기
    res["cohens_d"] = cohens_d(pos, neg)
    res["cliffs_delta"] = cliffs_delta(pos, neg)

    n_ov, rng = overlap_count(pos, neg)
    res["overlap_n"] = n_ov
    res["overlap_range"] = rng

    return res


# ==================================================================
# 메인
# ==================================================================
def main():
    ap = argparse.ArgumentParser(description="Step2b: G_p95 로 양성/음성 분리 검정")
    ap.add_argument("--weights", default=str(P.WEIGHTS_PATH))
    ap.add_argument("--solo_train_root", default=str(P.SOLO_TRAIN))
    ap.add_argument("--solo_test_root", default=str(P.SOLO_TEST))
    ap.add_argument("--out_dir", default=str(P.OUT_STEP2B))
    ap.add_argument("--iou", type=float, default=P.IOU)
    ap.add_argument("--imgsz", type=int, default=P.IMG_SIZE)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    CONF = P.CONF_MIN
    train_root, test_root = Path(args.solo_train_root), Path(args.solo_test_root)
    P.check(Path(args.weights), train_root, test_root)
    out_dir = P.ensure_dir(Path(args.out_dir))

    print("=" * 68)
    print("Step2b · G_p95 로 양성/음성이 분리되는가")
    print("=" * 68)
    print(f"  가중치 : {Path(args.weights).name}")
    print(f"  설정   : conf={CONF}, G채널 p95")
    print(f"  출력   : {out_dir}")
    print()

    model = YOLO(str(args.weights))
    names = model.model.names if hasattr(model.model, "names") else model.names
    roi_id = next(k for k, v in names.items() if str(v).lower() == "roi")

    imgs = list_images([train_root]) + list_images([test_root])
    train_resolved = train_root.resolve()
    rows = []

    for i, ip in enumerate(imgs, 1):
        lab = solo_label(ip)
        if lab is None:
            continue
        img = imread_unicode(ip)
        if img is None:
            continue

        r = model.predict(source=img, imgsz=args.imgsz, conf=CONF,
                          iou=args.iou, device=args.device, verbose=False)[0]
        rois = [b for b, c in zip(r.boxes.xyxy.cpu().numpy(),
                                  r.boxes.cls.cpu().numpy().astype(int))
                if c == roi_id]
        if not rois:
            continue

        best_v, best_sat = -1.0, np.nan
        for b in rois:
            v, sat = g_p95(safe_crop(img, b))
            if np.isfinite(v) and v > best_v:
                best_v, best_sat = v, sat

        rows.append({
            "image_id": ip.stem, "image_path": str(ip),
            "split": "train" if train_resolved in ip.resolve().parents else "test",
            "label": lab,
            "G_p95_au": f"{best_v:.4f}",
            "sat_frac": f"{best_sat:.4f}",
        })

        if i % 30 == 0 or i == len(imgs):
            print(f"  [{i}/{len(imgs)}]")

    if not rows:
        raise SystemExit("측정된 ROI 가 없습니다.")

    with open(out_dir / "Gp95_values.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def pick(split, label):
        return np.array([float(r["G_p95_au"]) for r in rows
                         if r["label"] == label
                         and (split is None or r["split"] == split)])

    results = []
    for split, name in ((None, "전체"), ("train", "train"), ("test", "test")):
        results.append(analyze(pick(split, "pos"), pick(split, "neg"), name))

    with open(out_dir / "Gp95_stats.csv", "w", newline="", encoding="utf-8-sig") as f:
        keys = sorted({k for r in results for k in r})
        w = csv.DictWriter(f, fieldnames=["split"] + [k for k in keys if k != "split"])
        w.writeheader()
        for r in results:
            w.writerow({k: (json.dumps(v, ensure_ascii=False)
                            if isinstance(v, (list, tuple)) else v)
                        for k, v in r.items()})

    (out_dir / "summary.json").write_text(
        json.dumps({"settings": {"conf": CONF, "method": "G", "metric": "p95"},
                    "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    # ---------------- 출력 ----------------
    for r in results:
        print()
        print("=" * 68)
        print(f"[{r['split']}]  양성 n={r['n_pos']}, 음성 n={r['n_neg']}")
        print("=" * 68)
        if r.get("note"):
            print(f"  {r['note']}")
            continue

        print(f"  양성  평균 {r['pos_mean']:7.2f}  SD {r['pos_sd']:6.2f}  "
              f"중앙값 {r['pos_median']:7.2f}")
        print(f"  음성  평균 {r['neg_mean']:7.2f}  SD {r['neg_sd']:6.2f}  "
              f"중앙값 {r['neg_median']:7.2f}")
        print()
        print(f"  정규성 (Shapiro-Wilk)")
        print(f"    양성  W={r['shapiro_pos_W']:.4f}  p={r['shapiro_pos_p']:.4f}  "
              f"{'정규' if r['shapiro_pos_p'] > 0.05 else '비정규'}")
        print(f"    음성  W={r['shapiro_neg_W']:.4f}  p={r['shapiro_neg_p']:.4f}  "
              f"{'정규' if r['shapiro_neg_p'] > 0.05 else '비정규'}")
        print()
        print(f"  Welch t-test      t={r['welch_t']:8.3f}   p={r['welch_p']:.3e}")
        print(f"  Mann-Whitney U    U={r['mannwhitney_U']:8.1f}   p={r['mannwhitney_p']:.3e}")
        if not r["both_normal"]:
            if r["same_conclusion"]:
                print("    → 정규성은 성립하지 않지만 두 검정의 결론이 같다.")
                print("      가정 위반이 결과를 뒤집지는 않았다.")
            else:
                print("    → 두 검정의 결론이 다르다. 비모수 결과를 따라야 한다.")
        print()
        print(f"  Cohen's d         {r['cohens_d']:.3f}")
        print(f"  Cliff's delta     {r['cliffs_delta']:.3f}"
              f"   (1 이면 완전 분리)")
        if r["overlap_n"] == 0:
            lo, hi = r["overlap_range"]
            print(f"  두 집단이 완전히 분리된다 "
                  f"(음성 최대 {lo:.1f} < 양성 최소 {hi:.1f})")
        else:
            lo, hi = r["overlap_range"]
            print(f"  겹치는 구간 [{lo:.1f}, {hi:.1f}] 에 {r['overlap_n']}개 표본")

    print()
    print(f"[저장] {out_dir}")


if __name__ == "__main__":
    main()