# -*- coding: utf-8 -*-
"""
step2a_channel_selection.py

목적
    ROI 의 형광 세기를 무엇으로 대표할지 정한다.

    색을 수치로 바꾸는 방법은 여럿이다. 어떤 채널을 쓰느냐, 그 안에서
    평균을 쓰느냐 상위값을 쓰느냐에 따라 양성과 음성의 구분력이 달라진다.
    7개 채널 × 2개 지표 = 14개 조합을 모두 계산하고, 양성과 음성을 가장
    크게 갈라놓는 조합을 고른다.

채널
    G        초록 채널 원값. FAM 형광의 발광 파장이 녹색 영역이다.
    HSV_V    명도. 색을 무시하고 밝기만 본다.
    GRAY     흑백 변환값.
    HSV_S    채도. 색이 얼마나 진한가.
    G_norm   G / (R+G+B). 전체 밝기로 나눠 조명 세기의 영향을 줄인다.
    G_ratio  G / (R+B). 초록이 다른 색 대비 얼마나 강한가.
    ExG      2G - R - B. 식물 영상에서 쓰는 초록 강조 지표.

지표
    mean     ROI 전체 픽셀의 평균
    p95      상위 5% 지점의 값

    ROI 박스 안에는 형광이 없는 배경 픽셀도 섞인다. 평균을 쓰면 배경이
    신호를 희석시키므로, 상위값이 더 안정적일 수 있다.

선정 기준
    Cohen's d — 두 집단 평균 차이를 표준편차로 나눈 값이다.
    단위가 다른 채널끼리도 비교할 수 있어 이런 선택에 적합하다.
    통상 0.8 이상이면 큰 효과로 본다.

출력
    results/step2a_channel/
      ├── channel_scan_values.csv    ROI 별 14개 조합 값
      ├── channel_scan_report.csv    조합별 Cohen's d, t검정, 포화 비율
      ├── best_channel.json          선정 결과
      └── viz/                       (--save_viz 지정 시)

실행
    python analysis/step2a_channel_selection.py

주의 · 포화(saturation)
    8비트 이미지의 채널값 상한은 255 다. 형광이 그보다 강하면 255 로 잘려
    실제 세기를 알 수 없게 된다. 포화된 값이 많으면 분산이 인위적으로
    줄어들어 Cohen's d 가 과대평가된다.
    따라서 조합별 포화 비율을 함께 계산해 보고한다.

원본 대비 수정 사항
    - pair 이미지의 아래쪽 ROI 를 무조건 "pos" 로 라벨링하고 있었다.
      neg_neg pair 를 넘기면 음성이 양성으로 잘못 들어간다.
      경로에 neg_neg 가 있으면 건너뛰도록 안전장치를 추가했다.
    - 하드코딩 경로 제거, 오버레이 저장 기본 끄기, 한글 경로 대응
    - Welch t검정과 포화 비율을 리포트에 추가

실행 결과 (2026-08, weights.pt, solo 110장)
    G · p95 선정.  Cohen's d = 3.191

        조합            d       95% CI          p          포화
        G_p95        3.191   [2.80, 3.81]   1.1e-31      3.5%
        G_mean       2.984   [2.64, 3.49]   3.6e-28      3.5%
        GRAY_p95     2.392   [2.09, 2.86]   7.3e-22      3.5%
        G_norm_p95   2.089   [1.60, 2.83]   9.3e-19      3.5%
        HSV_V_mean   0.776   [0.44, 1.11]   9.5e-05      3.5%
        G_ratio_p95  0.256  [-0.10, 0.66]   1.8e-01      3.5%
        HSV_S_p95   -0.318  [-0.64, 0.06]   1.0e-01      3.5%
        (전체 14개는 channel_scan_report.csv 참고)

    G 계열과 GRAY 계열은 신뢰구간이 거의 겹치지 않아 통계적으로 구분된다.
    FAM 형광의 발광 파장이 녹색 영역이라는 물리적 근거와도 일치한다.

    다만 같은 G 채널의 mean(d=2.98)과는 구간이 크게 겹쳐, p95 가 mean 보다
    낫다고 단언하기는 어렵다. p95 를 택한 근거는 통계적 우위가 아니라
    ROI 안의 배경 픽셀에 덜 희석된다는 점이다.

    조명 보정을 의도한 정규화 계열은 오히려 구분력이 낮았다.
    특히 G/(R+B) 는 p=0.18 로 유의하지 않다. 앰버 필터 때문에 R 채널에도
    신호가 실려, 분모에서 신호끼리 상쇄된 것으로 보인다.

    HSV_S 는 d 가 음수다. 형광이 강해지면 R·G·B 가 함께 올라가 흰색에
    가까워지므로 채도는 오히려 떨어진다.

    양성 ROI 픽셀의 3.5% 가 포화(255) 상태다. p95 는 상위 5% 지점이므로
    아직 영향권 밖이지만 여유가 크지 않다.
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
METHODS = ["G", "HSV_V", "GRAY", "HSV_S", "G_norm", "G_ratio", "ExG"]
SATURATION_LEVEL = 254.0        # 이 값 이상이면 포화로 본다

VIZ_MAX_WIDTH = 1024
VIZ_JPEG_QUALITY = 85


# ==================================================================
# 유틸
# ==================================================================
def list_images(roots):
    paths = []
    for r in roots:
        r = Path(r)
        if r.is_file() and r.suffix.lower() in IMG_EXTS:
            paths.append(r)
        elif r.is_dir():
            paths.extend(p for p in r.rglob("*") if p.suffix.lower() in IMG_EXTS)
    return sorted(set(paths))


def imread_unicode(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def to_xyxy(b):
    return [int(float(v)) for v in b[:4]]


def inside(inner, outer):
    x1, y1, x2, y2 = inner
    X1, Y1, X2, Y2 = outer
    return x1 >= X1 and y1 >= Y1 and x2 <= X2 and y2 <= Y2


def center_y(b):
    return (b[1] + b[3]) / 2.0


def safe_crop(img, xyxy):
    if img is None or xyxy is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    H, W = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def norm_path(p) -> str:
    return str(p).lower().replace("\\", "/")


def is_pair(path) -> bool:
    return "/pair/" in norm_path(path)


def is_negneg(path) -> bool:
    return "/pair/neg_neg" in norm_path(path)


def is_solo(path) -> bool:
    return "/solo/" in norm_path(path)


def solo_label(path):
    low = norm_path(path)
    if "/pos/" in low:
        return "pos"
    if "/neg/" in low:
        return "neg"
    return None


# ==================================================================
# 채널 변환
# ==================================================================
def compute_value(method: str, crop_bgr):
    """
    crop 을 지정한 방법으로 단일 채널 맵으로 바꾼 뒤 mean 과 p95 를 낸다.
    비율 계열은 0~255 범위로 맞춰 다른 채널과 비교 가능하게 한다.
    """
    if crop_bgr is None:
        return None
    eps = 1e-6
    B, G, R = cv2.split(crop_bgr)
    B, G, R = B.astype(np.float32), G.astype(np.float32), R.astype(np.float32)

    if method == "G":
        M = G
    elif method == "HSV_V":
        M = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)[:, :, 2].astype(np.float32)
    elif method == "GRAY":
        M = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    elif method == "HSV_S":
        M = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    elif method == "G_norm":
        M = G / (R + G + B + eps) * 255.0
    elif method == "G_ratio":
        M = G / (R + B + eps)
        M = np.clip(M, 0, np.percentile(M, 99.9))
        M = M / (np.max(M) + eps) * 255.0
    elif method == "ExG":
        M = 2.0 * G - R - B
        lo, hi = np.percentile(M, 1.0), np.percentile(M, 99.0)
        if hi <= lo:
            hi = lo + 1.0
        M = np.clip((M - lo) / (hi - lo), 0.0, 1.0) * 255.0
    else:
        return None

    return {
        "mean_au": float(np.mean(M)),
        "p95_au": float(np.percentile(M, 95.0)),
        # 원본 채널이 포화됐는지는 G 원값 기준으로 본다
        "sat_frac": float(np.mean(G >= SATURATION_LEVEL)),
    }


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


def bootstrap_d_ci(pos, neg, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 3 or len(neg) < 3:
        return None
    out = []
    for _ in range(n):
        d = cohens_d(rng.choice(pos, len(pos), replace=True),
                     rng.choice(neg, len(neg), replace=True))
        if np.isfinite(d):
            out.append(d)
    if len(out) < n * 0.5:
        return None
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"lo": float(lo), "hi": float(hi)}


def save_viz(img, tubes, rois, out_path: Path):
    draw = img.copy()
    for b in tubes:
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 3)
    for b in rois:
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (255, 0, 255), 3)
    h, w = draw.shape[:2]
    if w > VIZ_MAX_WIDTH:
        s = VIZ_MAX_WIDTH / w
        draw = cv2.resize(draw, (VIZ_MAX_WIDTH, int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path.with_suffix(".jpg")), draw,
                [cv2.IMWRITE_JPEG_QUALITY, VIZ_JPEG_QUALITY])


# ==================================================================
# 메인
# ==================================================================
def main():
    ap = argparse.ArgumentParser(description="Step2a: 형광 대표값 채널·지표 선정")
    ap.add_argument("--weights", default=str(P.WEIGHTS_PATH))
    ap.add_argument("--roots", nargs="*", default=None,
                    help="생략하면 solo train/test 를 쓴다")
    ap.add_argument("--out_dir", default=str(P.OUT_STEP2A))
    ap.add_argument("--iou", type=float, default=P.IOU)
    ap.add_argument("--imgsz", type=int, default=P.IMG_SIZE)
    ap.add_argument("--device", default="")
    ap.add_argument("--max_tubes", type=int, default=2,
                    help="conf 상위 N개 튜브만 고려")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--save_viz", action="store_true")
    args = ap.parse_args()

    CONF = P.CONF_MIN
    roots = [Path(r) for r in args.roots] if args.roots else [P.SOLO_TRAIN, P.SOLO_TEST]
    P.check(Path(args.weights), *roots)

    out_dir = P.ensure_dir(Path(args.out_dir))
    viz_dir = P.ensure_dir(out_dir / "viz") if args.save_viz else None

    print("=" * 68)
    print("Step2a · 형광 대표값 채널·지표 선정")
    print("=" * 68)
    print(f"  가중치 : {Path(args.weights).name}")
    print(f"  설정   : conf={CONF}")
    print(f"  조합   : {len(args.methods)}채널 × 2지표 = {len(args.methods)*2}개")
    print(f"  출력   : {out_dir}")
    print()

    model = YOLO(str(args.weights))
    names = model.model.names if hasattr(model.model, "names") else model.names
    tube_id = next(k for k, v in names.items() if str(v).lower() == "tube")
    roi_id = next(k for k, v in names.items() if str(v).lower() == "roi")

    imgs = [p for p in list_images(roots) if "test_all" not in norm_path(p)]
    if not imgs:
        raise SystemExit("이미지를 찾지 못했습니다.")
    print(f"[INFO] 대상 {len(imgs)}장 (test_all 은 제외)\n")

    rows = []
    n_skip_negneg = 0

    for i, ip in enumerate(imgs, 1):
        # neg_neg pair 는 아래쪽이 음성이므로 "아래=pos" 규칙이 성립하지 않는다
        if is_negneg(ip):
            n_skip_negneg += 1
            continue

        img = imread_unicode(ip)
        if img is None:
            print(f"  [WARN] 읽기 실패: {ip.name}")
            continue

        r = model.predict(source=img, conf=CONF, iou=args.iou,
                          imgsz=args.imgsz, device=args.device, verbose=False)[0]

        tubes, tconf, rois, rconf = [], [], [], []
        for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(),
                            r.boxes.cls.cpu().numpy().astype(int),
                            r.boxes.conf.cpu().numpy()):
            if c == tube_id:
                tubes.append(to_xyxy(b)); tconf.append(float(cf))
            elif c == roi_id:
                rois.append(to_xyxy(b)); rconf.append(float(cf))

        # conf 상위 N개 튜브만
        order = sorted(range(len(tubes)), key=lambda k: tconf[k], reverse=True)
        sel = order[: min(args.max_tubes, len(order))]

        # 각 튜브 안에 완전히 포함된 ROI 중 conf 최고 1개
        pairs = []
        for k in sel:
            tb = tubes[k]
            contained = [(rb, rc) for rb, rc in zip(rois, rconf) if inside(rb, tb)]
            if contained:
                contained.sort(key=lambda x: x[1], reverse=True)
                pairs.append((tb, tconf[k], contained[0][0], contained[0][1]))

        assigned = []
        if is_pair(ip):
            tri = sorted(pairs, key=lambda x: center_y(x[2]))
            if len(tri) >= 1:
                assigned.append(("neg", tri[0]))     # 위 = NC
            if len(tri) >= 2:
                assigned.append(("pos", tri[1]))     # 아래 = 시료
        elif is_solo(ip):
            lab = solo_label(ip)
            if lab:
                assigned = [(lab, pr) for pr in pairs]

        for lab, (tb, tcf, rb, rcf) in assigned:
            crop = safe_crop(img, rb)
            if crop is None:
                continue
            for m in args.methods:
                v = compute_value(m, crop)
                if v is None:
                    continue
                rows.append({
                    "image_id": ip.stem, "image_path": str(ip),
                    "role": "pair" if is_pair(ip) else "solo",
                    "label": lab, "method": m,
                    "tube_conf": f"{tcf:.4f}", "roi_conf": f"{rcf:.4f}",
                    "mean_au": f"{v['mean_au']:.6f}",
                    "p95_au": f"{v['p95_au']:.6f}",
                    "sat_frac": f"{v['sat_frac']:.6f}",
                })

        if viz_dir is not None:
            save_viz(img, tubes, rois, viz_dir / ip.stem)
        if i % 30 == 0 or i == len(imgs):
            print(f"  [{i}/{len(imgs)}]")

    if n_skip_negneg:
        print(f"\n[INFO] neg_neg pair {n_skip_negneg}장 제외 "
              f"(아래쪽이 음성이라 pair 라벨 규칙과 맞지 않음)")

    if not rows:
        raise SystemExit("측정된 ROI 가 없습니다.")

    with open(out_dir / "channel_scan_values.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---------------- 조합별 평가 ----------------
    report = []
    for m in args.methods:
        sub = [r for r in rows if r["method"] == m]
        for metric in ("mean_au", "p95_au"):
            pos = np.array([float(r[metric]) for r in sub if r["label"] == "pos"])
            neg = np.array([float(r[metric]) for r in sub if r["label"] == "neg"])
            if len(pos) < 2 or len(neg) < 2:
                continue
            d = cohens_d(pos, neg)
            ci = bootstrap_d_ci(pos, neg)
            t, p = stats.ttest_ind(pos, neg, equal_var=False)
            sat = np.mean([float(r["sat_frac"]) for r in sub if r["label"] == "pos"])
            report.append({
                "method": m, "metric": metric.replace("_au", ""),
                "cohens_d": d,
                "d_ci_lo": ci["lo"] if ci else None,
                "d_ci_hi": ci["hi"] if ci else None,
                "welch_t": float(t), "welch_p": float(p),
                "pos_mean": float(np.mean(pos)), "neg_mean": float(np.mean(neg)),
                "pos_n": len(pos), "neg_n": len(neg),
                "pos_sat_frac": float(sat),
            })

    report.sort(key=lambda r: (-r["cohens_d"] if np.isfinite(r["cohens_d"]) else 1e9))

    with open(out_dir / "channel_scan_report.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader(); w.writerows(report)

    best = report[0]
    (out_dir / "best_channel.json").write_text(json.dumps({
        "best": {"method": best["method"], "metric": best["metric"],
                 "cohens_d": best["cohens_d"]},
        "all": report,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------- 출력 ----------------
    print()
    print("=" * 68)
    print("Cohen's d 순위")
    print("=" * 68)
    print(f"  {'조합':16s} {'d':>7s}  {'95% CI':>18s}  {'p':>10s}  {'포화':>6s}")
    print("  " + "-" * 64)
    for r in report:
        ci = (f"[{r['d_ci_lo']:.2f}, {r['d_ci_hi']:.2f}]"
              if r["d_ci_lo"] is not None else "—")
        mark = " ←" if r is best else ""
        print(f"  {r['method']+'_'+r['metric']:16s} {r['cohens_d']:7.3f}  "
              f"{ci:>18s}  {r['welch_p']:10.2e}  {r['pos_sat_frac']*100:5.1f}%{mark}")

    print()
    print("=" * 68)
    print(f"  선정: {best['method']} · {best['metric']}   Cohen's d = {best['cohens_d']:.3f}")
    print(f"        양성 평균 {best['pos_mean']:.2f} (n={best['pos_n']})")
    print(f"        음성 평균 {best['neg_mean']:.2f} (n={best['neg_n']})")

    # G_p95 와의 비교
    gp95 = next((r for r in report
                 if r["method"] == "G" and r["metric"] == "p95"), None)
    if gp95 and gp95 is not best:
        print(f"\n  참고: 현재 시스템이 쓰는 G · p95 는 d = {gp95['cohens_d']:.3f} 로 "
              f"{report.index(gp95)+1}위")
        if (best["d_ci_lo"] is not None and gp95["cohens_d"] >= best["d_ci_lo"]):
            print("        선정 조합의 신뢰구간 안에 들어오므로 "
                  "두 조합의 차이는 통계적으로 뚜렷하지 않다.")
    elif gp95 is best:
        print("\n  현재 시스템이 쓰는 조합과 일치한다.")

    if best["pos_sat_frac"] > 0.01:
        print(f"\n  [주의] 양성 ROI 픽셀의 {best['pos_sat_frac']*100:.1f}% 가 포화 상태다.")
        print("         포화가 많으면 분산이 줄어 Cohen's d 가 과대평가된다.")

    print("=" * 68)
    print(f"\n[저장] {out_dir}")


if __name__ == "__main__":
    main()