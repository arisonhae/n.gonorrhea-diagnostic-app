# -*- coding: utf-8 -*-
"""
negneg_false_positive_check.py

배경
    이 시스템은 위 튜브에 음성 대조군(NC), 아래 튜브에 검사 시료를 놓고
    ratio = I_sample / I_nc 로 판정한다.

    아래에도 음성을 놓으면 두 튜브의 형광이 같아야 하므로 ratio 는 1 근처가
    되어야 하고, 임계값을 넘어서는 안 된다. 넘는다면 그것이 위양성이다.

    pair/neg_neg 26장은 논문 제출 이후(2025-11-17)에 이 확인을 위해 촬영했다.

step4 와의 관계
    step4 도 neg_neg 를 읽어 위양성 개수를 세지만, 그것은 임계값을 정하는
    과정의 부산물이다. 이 스크립트는 위양성 자체에 집중한다.

        · 위양성률과 그 신뢰구간
        · 어떤 이미지가 오판되는지, 그 이미지의 특징은 무엇인지
        · ratio 분포가 1 을 중심으로 대칭인지, 한쪽으로 치우쳤는지
        · 위·아래 튜브 중 어느 쪽이 흔들려서 ratio 가 커지는지

    마지막 항목이 중요하다. ratio 가 커지는 경로는 두 가지다.
    시료가 밝아지거나(분자), NC 가 어두워지거나(분모).
    후자라면 원인이 시료가 아니라 촬영이나 NC 자체에 있다는 뜻이다.

출력
    results/negneg_check/
      ├── negneg_analysis.csv     이미지별 I_nc, I_sample, ratio, 판정
      ├── false_positives.csv     오판된 이미지만
      └── summary.json

실행
    python analysis/negneg_false_positive_check.py

    # 오판된 이미지의 검출 결과를 눈으로 확인
    python analysis/negneg_false_positive_check.py --save_viz fp

실행 결과 (2026-08, weights.pt, T_ratio=1.1162)

    위양성률 16.7%  (4/24)   95% CI 6.7 – 35.9%

    표본이 24장뿐이라 구간이 매우 넓다. 실제 위양성률은 7% 일 수도,
    36% 일 수도 있다. 점추정값만 인용하면 안 되는 사례다.

    ratio 분포
        음성-음성  평균 1.0226  SD 0.0887  중앙값 0.9898  범위 [0.899, 1.217]
        음성-양성  평균 1.2539                          범위 [1.116, 1.362]

    두 튜브가 모두 음성이면 ratio 는 1.0 이어야 한다. 실제 평균은 1.0226 으로
    +2.3% 벗어나 있으나 p=0.224 로 유의하지 않다. 체계적 편향은 없고,
    문제는 개별 촬영의 산포(SD 0.089)다.

    두 분포는 구간 [1.1157, 1.2174] 에서 겹치며, 여기에 음성-음성 4장이
    들어 있다. 임계값을 어디에 두어도 이 겹침은 사라지지 않는다.

    오판된 이미지 (전체 중앙값: NC 202.5, 시료 203.0)

        이미지        ratio    I_nc   I_sample
        neg_neg_21   1.1553   161.0    186.0
        neg_neg_22   1.1749   183.0    215.0
        neg_neg_23   1.2174   184.0    224.0
        neg_neg_24   1.1889   180.0    214.0

    네 장 모두 I_nc 가 중앙값(202.5)보다 낮다. 예외가 없다.
    즉 시료가 밝아서가 아니라 기준 튜브가 어둡게 찍혀 생긴 오판이다.
    검출 결과 이미지를 확인한 결과 ROI 검출 자체는 정상이었다.

    임계값을 올려도 해결되지 않는다
        T=1.1162   위양성  4/24   위음성  1/20   ← 현재
        T=1.1500   위양성  4/24   위음성  2/20
        T=1.2000   위양성  1/24   위음성  5/20
        T=1.2175   위양성  0/24   위음성  9/20

    위양성 4건을 없애려면 위음성이 1 에서 9 로 늘어난다.
    진단에서는 감염자를 놓치는 쪽이 더 위험하므로 현재 값이 합리적이다.

    근본 원인은 측정 변동이다. 같은 음성 시료라도 두 튜브의 측정값이
    ±9% 정도 흔들리며(SD 0.089), 그 변동이 판정 마진(약 11.6%)과
    비슷한 크기다. 임계값 조정이나 코드 수정으로 해결되지 않으며,
    두 튜브가 같은 조명 조건에 놓이도록 촬영 방식을 바꾸거나
    표본을 늘려 분포를 더 정확히 파악해야 한다.
"""

import argparse
import csv
import json
import sys
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
VIZ_MAX_WIDTH = 1024
VIZ_JPEG_QUALITY = 85


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
        return np.nan, np.nan
    g = crop[:, :, 1].astype(np.float32)
    if g.size == 0:
        return np.nan, np.nan
    return float(np.percentile(g, 95.0)), float(np.mean(g >= SATURATION_LEVEL))


def center_y(b):
    return (float(b[1]) + float(b[3])) / 2.0


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    s = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return {"point": p, "lo": float((c - s) / d), "hi": float((c + s) / d),
            "k": int(k), "n": int(n)}


def save_viz(img, tubes, rois, out_path: Path, note=""):
    draw = img.copy()
    for b in tubes:
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
    for j, b in enumerate(rois):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(draw, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(draw, "NC" if j == 0 else "SAMPLE", (x1, max(24, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2, cv2.LINE_AA)
    if note:
        cv2.putText(draw, note, (12, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 3, cv2.LINE_AA)
    h, w = draw.shape[:2]
    if w > VIZ_MAX_WIDTH:
        s = VIZ_MAX_WIDTH / w
        draw = cv2.resize(draw, (VIZ_MAX_WIDTH, int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path.with_suffix(".jpg")), draw,
                [cv2.IMWRITE_JPEG_QUALITY, VIZ_JPEG_QUALITY])


def main():
    ap = argparse.ArgumentParser(description="음성-음성 쌍의 위양성 확인")
    ap.add_argument("--weights", default=str(P.WEIGHTS_PATH))
    ap.add_argument("--negneg_root", default=str(P.PAIR_NEGNEG))
    ap.add_argument("--negpos_root", default=str(P.PAIR_NEGPOS),
                    help="비교용. 양성 쌍의 ratio 분포와 대조한다")
    ap.add_argument("--out_dir", default=str(P.OUT_NEGNEG))
    ap.add_argument("--t_ratio", type=float, default=P.RATIO_THR)
    ap.add_argument("--iou", type=float, default=P.IOU)
    ap.add_argument("--imgsz", type=int, default=P.IMG_SIZE)
    ap.add_argument("--device", default="")
    ap.add_argument("--save_viz", choices=["off", "fp", "all"], default="off",
                    help="fp 는 오판된 이미지만 저장한다")
    args = ap.parse_args()

    CONF = P.CONF_MIN
    out_dir = P.ensure_dir(Path(args.out_dir))
    viz_dir = P.ensure_dir(out_dir / "viz") if args.save_viz != "off" else None
    P.check(Path(args.weights), Path(args.negneg_root))

    print("=" * 66)
    print("음성-음성 쌍의 위양성 확인")
    print("=" * 66)
    print(f"  가중치   : {Path(args.weights).name}")
    print(f"  임계값   : {args.t_ratio}")
    print(f"  출력     : {out_dir}")
    print()

    model = YOLO(str(args.weights))
    names = model.model.names if hasattr(model.model, "names") else model.names
    tube_id = next(k for k, v in names.items() if str(v).lower() == "tube")
    roi_id = next(k for k, v in names.items() if str(v).lower() == "roi")

    def measure(ip: Path, group: str):
        img = imread_unicode(ip)
        if img is None:
            return None, None, None, None
        r = model.predict(source=img, imgsz=args.imgsz, conf=CONF,
                          iou=args.iou, device=args.device, verbose=False)[0]
        tubes, rois = [], []
        for b, c in zip(r.boxes.xyxy.cpu().numpy(),
                        r.boxes.cls.cpu().numpy().astype(int)):
            (tubes if c == tube_id else rois).append(b)
        rs = sorted(rois, key=center_y)
        if len(rs) < 2:
            return img, tubes, rs, {"note": "ROI_PARTIAL" if len(rs) == 1 else "ROI_NONE"}

        Iu, su = g_p95(safe_crop(img, rs[0]))
        Il, sl = g_p95(safe_crop(img, rs[1]))
        if not (np.isfinite(Iu) and np.isfinite(Il) and Iu > 0):
            return img, tubes, rs, {"note": "MEASURE_FAIL"}
        return img, tubes, rs, {
            "I_nc": Iu, "I_sample": Il, "ratio": Il / Iu,
            "sat_nc": su, "sat_sample": sl, "note": "",
        }

    rows = []
    for group, root in (("neg_neg", args.negneg_root), ("neg_pos", args.negpos_root)):
        if not Path(root).exists():
            continue
        imgs = list_images(root)
        print(f"  {group}: {len(imgs)}장")
        for ip in imgs:
            img, tubes, rs, m = measure(ip, group)
            if m is None:
                continue
            row = {"group": group, "image_id": ip.stem, "image_path": str(ip)}
            if m["note"]:
                row.update({"I_nc": "", "I_sample": "", "ratio": "",
                            "sat_nc": "", "sat_sample": "",
                            "pred": "", "correct": "", "note": m["note"]})
            else:
                pred = "pos" if m["ratio"] >= args.t_ratio else "neg"
                truth = "neg" if group == "neg_neg" else "pos"
                row.update({
                    "I_nc": f"{m['I_nc']:.4f}", "I_sample": f"{m['I_sample']:.4f}",
                    "ratio": f"{m['ratio']:.6f}",
                    "sat_nc": f"{m['sat_nc']:.4f}", "sat_sample": f"{m['sat_sample']:.4f}",
                    "pred": pred, "correct": str(pred == truth), "note": "",
                })
                if viz_dir is not None:
                    is_fp = (group == "neg_neg" and pred == "pos")
                    if args.save_viz == "all" or (args.save_viz == "fp" and is_fp):
                        save_viz(img, tubes, rs, viz_dir / f"{group}__{ip.stem}",
                                 note=f"ratio={m['ratio']:.3f}" + (" FP" if is_fp else ""))
            rows.append(row)

    with open(out_dir / "negneg_analysis.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    nn = [r for r in rows if r["group"] == "neg_neg" and r["ratio"]]
    npos = [r for r in rows if r["group"] == "neg_pos" and r["ratio"]]
    if not nn:
        raise SystemExit("유효한 neg_neg 데이터가 없습니다.")

    nn_ratio = np.array([float(r["ratio"]) for r in nn])
    nn_nc = np.array([float(r["I_nc"]) for r in nn])
    nn_sm = np.array([float(r["I_sample"]) for r in nn])
    fps = [r for r in nn if r["pred"] == "pos"]

    # ---------------- 위양성률 ----------------
    print()
    print("=" * 66)
    print("1. 위양성률")
    print("=" * 66)
    ci = wilson_ci(len(fps), len(nn))
    print(f"  음성-음성 쌍 {len(nn)}장 중 {len(fps)}장이 양성으로 오판")
    print(f"  위양성률 {ci['point']*100:.1f}%   "
          f"95% CI {ci['lo']*100:.1f}–{ci['hi']*100:.1f}%")
    print(f"  → 실제로는 {ci['lo']*100:.0f}% 에서 {ci['hi']*100:.0f}% 사이일 수 있다.")

    # ---------------- 분포 ----------------
    print()
    print("=" * 66)
    print("2. ratio 분포")
    print("=" * 66)
    print(f"  음성-음성  평균 {nn_ratio.mean():.4f}  SD {nn_ratio.std(ddof=1):.4f}  "
          f"중앙값 {np.median(nn_ratio):.4f}")
    print(f"             범위 [{nn_ratio.min():.4f}, {nn_ratio.max():.4f}]")
    print(f"  두 튜브가 같은 음성이므로 이론적으로 1.0 이어야 한다.")
    dev = (nn_ratio.mean() - 1.0) * 100
    print(f"  실제 평균은 1.0 에서 {dev:+.1f}% 벗어나 있다.")

    t1, p1 = stats.ttest_1samp(nn_ratio, 1.0)
    print(f"  1.0 과의 차이  t={t1:.3f}  p={p1:.4f}  "
          f"{'유의하다' if p1 < 0.05 else '유의하지 않다'}")

    if npos:
        np_ratio = np.array([float(r["ratio"]) for r in npos])
        print(f"\n  음성-양성  평균 {np_ratio.mean():.4f}  "
              f"범위 [{np_ratio.min():.4f}, {np_ratio.max():.4f}]")
        gap = np_ratio.min() - nn_ratio.max()
        if gap > 0:
            print(f"  두 분포가 분리된다 (간격 {gap:.4f})")
        else:
            n_ov = int(np.sum(nn_ratio >= np_ratio.min()))
            print(f"  두 분포가 겹친다. 겹치는 구간 "
                  f"[{np_ratio.min():.4f}, {nn_ratio.max():.4f}] 에 "
                  f"음성-음성 {n_ov}장")
            print(f"  임계값을 어디에 두어도 이 겹침은 사라지지 않는다.")

    # ---------------- 오판 원인 ----------------
    print()
    print("=" * 66)
    print("3. 오판된 이미지")
    print("=" * 66)
    if not fps:
        print("  없음")
    else:
        med_nc, med_sm = np.median(nn_nc), np.median(nn_sm)
        print(f"  전체 중앙값:  NC {med_nc:.1f}   시료 {med_sm:.1f}\n")
        print(f"  {'이미지':28s} {'ratio':>8s} {'I_nc':>8s} {'I_sample':>10s}   원인")
        print("  " + "-" * 62)
        causes = {"nc_low": 0, "sample_high": 0, "both": 0}
        for r in sorted(fps, key=lambda x: -float(x["ratio"])):
            nc, sm = float(r["I_nc"]), float(r["I_sample"])
            nc_low = nc < med_nc
            sm_high = sm > med_sm
            if nc_low and sm_high:
                cause = "NC 어둡고 시료 밝음"; causes["both"] += 1
            elif nc_low:
                cause = "NC 가 어두움"; causes["nc_low"] += 1
            elif sm_high:
                cause = "시료가 밝음"; causes["sample_high"] += 1
            else:
                cause = "—"
            print(f"  {r['image_id'][:28]:28s} {float(r['ratio']):8.4f} "
                  f"{nc:8.1f} {sm:10.1f}   {cause}")

        print()
        if causes["nc_low"] + causes["both"] > causes["sample_high"]:
            print("  분모(NC)가 어두워진 경우가 더 많다.")
            print("  즉 시료가 밝아서가 아니라 기준 튜브가 어둡게 찍혀 생긴 오판이다.")
            print("  촬영 시 두 튜브가 같은 조명을 받도록 하는 것이 중요하다.")
        elif causes["sample_high"] > 0:
            print("  분자(시료)가 밝아진 경우가 더 많다.")
            print("  음성 시료에서 배경 형광이 높게 나온 것으로,")
            print("  반응 자체의 배경 신호를 확인할 필요가 있다.")

        with open(out_dir / "false_positives.csv", "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(fps[0].keys()))
            w.writeheader(); w.writerows(fps)

    # ---------------- 임계값을 올리면 ----------------
    print()
    print("=" * 66)
    print("4. 임계값을 올리면 위양성이 사라지는가")
    print("=" * 66)
    if npos:
        np_ratio = np.array([float(r["ratio"]) for r in npos])
        for T in (args.t_ratio, 1.15, 1.20, 1.25, float(nn_ratio.max()) + 1e-4):
            fp = int(np.sum(nn_ratio >= T))
            fn = int(np.sum(np_ratio < T))
            tag = "  ← 현재" if abs(T - args.t_ratio) < 1e-9 else ""
            print(f"  T={T:7.4f}   위양성 {fp:2d}/{len(nn_ratio)}   "
                  f"위음성 {fn:2d}/{len(np_ratio)}{tag}")
        print()
        print("  위양성을 없애려면 위음성을 감수해야 한다.")
        print("  진단에서는 감염자를 놓치는 쪽이 더 위험하므로,")
        print("  임계값을 올려 위양성을 없애는 것이 항상 옳지는 않다.")

    summary = {
        "settings": {"conf": CONF, "t_ratio": args.t_ratio},
        "false_positive_rate": ci,
        "negneg_ratio": {
            "n": len(nn_ratio), "mean": float(nn_ratio.mean()),
            "sd": float(nn_ratio.std(ddof=1)), "median": float(np.median(nn_ratio)),
            "min": float(nn_ratio.min()), "max": float(nn_ratio.max()),
            "ttest_vs_1_p": float(p1),
        },
        "false_positive_images": [r["image_id"] for r in fps],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[저장] {out_dir}")


if __name__ == "__main__":
    main()