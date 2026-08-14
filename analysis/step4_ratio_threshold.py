# -*- coding: utf-8 -*-
"""
step4_ratio_threshold.py

목적
    pair 이미지(위=NC, 아래=시료)에서 비율 기반 판정 임계값 T_ratio 를 도출한다.
    한 장에 두 튜브를 함께 찍으면 조명·기기 차이가 상쇄되므로,
    절댓값 대신 ratio = I_sample / I_nc 를 판정에 쓴다.

임계값 도출 방법 두 가지를 모두 계산하고 비교한다.

    [A-1] 전이(transfer) · neg_pos 만  — 2025-11-06 원본 도출 조건
          T_ratio = step3_cutoff / median(neg_pos 의 NC)
          solo 110장으로 구한 절댓값 기준을 pair 좌표계로 옮긴 것.
          앱에 박혀 있는 RATIO_THR = 1.148 이 이 값이다.

    [A-2] 전이 · neg_pos + neg_neg  — 확장 조건
          neg_neg pair 는 11-17 에 추가 촬영한 것으로, step4 를 처음
          작성한 시점에는 존재하지 않았다. 이 pair 의 위쪽 튜브도 NC 이므로
          기준 분포에 포함할 수 있고, 표본이 늘어 중앙값이 안정적이다.

    [B]   Youden's J  — pair ratio 분포 직접 최적화
          민감도+특이도가 최대가 되는 지점. pair 데이터에 직접 근거하지만
          표본이 적어 흔들린다. 최적점의 FPR 은 신호 겹침의 하한을 뜻한다.

    세 값 모두 부트스트랩 신뢰구간을 함께 산출해 서로 구분되는지 확인한다.

출력
    results/step4_ratio/
      ├── solo_analysis.csv          solo 이미지별 I
      ├── pair_analysis.csv          pair 이미지별 I_nc, I_sample, ratio
      ├── threshold_derivation.json  A/B 두 방식의 임계값과 CI
      ├── test_all_eval.csv          기기별 최종 검증 결과
      └── summary.json

실행
    python analysis/step4_ratio_threshold.py

원본 대비 수정 사항
    1. 위쪽 튜브 판정이 ("neg" 로) 하드코딩되어 있어 upper_acc 가 무의미했다.
       → 절대 cutoff 로 위쪽도 실제 판정하도록 변경.
       다만 test_all 에 위쪽이 양성인 이미지가 없어, 수정 후에도 이 경로는
       검증되지 않는다. 해당 상황이면 출력에 그 사실을 명시한다.
    5. 포화(G채널 255) 여부를 기록하고 경고한다.
       포화되면 실제 형광 세기를 알 수 없으므로 ratio 가 과소평가된다.
    2. 보정 방식(none/ratio/shift/affine)을 test_all 정확도로 골랐다.
       최종 검증용 데이터로 파라미터를 선택하면 성능이 부풀려진다.
       → 임계값 도출은 pair 데이터로만, test_all 은 평가 전용으로 분리.
    3. 판정이 delta/ratio/abs 의 OR 조합이었으나 실제 앱은 ratio 단독을 쓴다.
       → ratio 단독을 기본으로 하고 나머지는 비교용 지표로만 기록.
    4. 하드코딩 경로 제거, 오버레이 저장 기본 끄기, 한글 경로 대응.

실행 결과 (2026-08, weights.pt)

    T_ratio = 1.1162 채택 (A-2)

        방식                    n    median(NC)    T        95% CI
        A-1  neg_pos 만        20      192.50   1.1481  [1.1078, 1.2738]
        A-2  neg_pos+neg_neg   44      198.00   1.1162  [1.0914, 1.1571]
        B    Youden's J        44        —      1.1024  [1.0867, 1.1980]

    A-1 은 2025-11-06 원본 도출 조건이며, 앱에 있던 1.148 이 정확히
    재현되었다. neg_neg pair 는 11-17 에 추가 촬영한 것이라 원본 시점에는
    존재하지 않았다.

    A-2 를 채택한 이유는 두 가지다.
      · 신뢰구간 폭이 0.066 으로 A-1(0.166)의 절반 이하다.
        n=20 에서 중앙값은 리샘플링할 때마다 크게 흔들린다.
      · pair 자체 성능에서 위양성 수는 같고 위음성만 하나 적다.
        1.1481 → TP18 FN2 FP4 TN20  (86.4%)
        1.1162 → TP19 FN1 FP4 TN20  (88.6%)
        감염자를 놓치는 쪽이 더 위험하므로 낮은 임계값이 적절하다.

    Youden's J 의 FPR 이 16.7% 다. 최적점에서도 이 값이므로 임계값 조정으로
    위양성을 더 줄일 수 없다. neg_neg 24장 중 4장이 양성으로 오판되며,
    음성-음성 쌍과 음성-양성 쌍의 ratio 분포가 실제로 겹쳐 있다는 뜻이다.
    이 수치는 논문에 없다.

    기기별 NC 형광값
        Galaxy Note 8   209.2
        iPhone 13       201.7
        iPhone 13 Pro   176.0

    최대 약 19% 차이로 판정 마진(약 11.6%)보다 크다. 절댓값으로 판정했다면
    iPhone 13 Pro 는 전부 음성으로 나왔을 것이다. 비율 정규화를 쓰는 이유가
    데이터로 확인된다.

    test_all 위쪽 정확도 100% 는 검증된 값이 아니다. 평가 대상 13장이 모두
    위쪽 음성이라, 위쪽을 항상 음성이라고 답해도 같은 값이 나온다.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import paths as P

try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit("ultralytics 가 필요합니다:  pip install ultralytics") from e


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIZ_MAX_WIDTH = 1024
VIZ_JPEG_QUALITY = 85
N_BOOTSTRAP = 2000
RNG_SEED = 0


# ==================================================================
# 기본 유틸
# ==================================================================
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


SATURATION_LEVEL = 254.0


def g_p95_intensity(crop_bgr) -> float:
    if crop_bgr is None:
        return np.nan
    g = crop_bgr[:, :, 1].astype(np.float32)
    return float(np.percentile(g, 95.0)) if g.size else np.nan


def sat_frac(crop_bgr) -> float:
    """포화(255 근처)된 픽셀의 비율. 이 값이 5%%를 넘으면 p95 자체가 포화된다."""
    if crop_bgr is None:
        return np.nan
    g = crop_bgr[:, :, 1].astype(np.float32)
    return float(np.mean(g >= SATURATION_LEVEL)) if g.size else np.nan


def center_y(b):
    return (float(b[1]) + float(b[3])) / 2.0


def solo_label_from_path(p: Path):
    low = str(p).lower().replace("\\", "/")
    if "/pos/" in low:
        return "pos"
    if "/neg/" in low:
        return "neg"
    return None


def save_viz(img, tubes, rois, out_path: Path):
    draw = img.copy()
    for b in tubes:
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
    for b in rois:
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(draw, (x1, y1), (x2, y2), (255, 0, 255), 3)
    h, w = draw.shape[:2]
    if w > VIZ_MAX_WIDTH:
        s = VIZ_MAX_WIDTH / w
        draw = cv2.resize(draw, (VIZ_MAX_WIDTH, int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path.with_suffix(".jpg")), draw,
                [cv2.IMWRITE_JPEG_QUALITY, VIZ_JPEG_QUALITY])


# ==================================================================
# test_all 파일명에서 기대 결과 읽기
# ==================================================================
def expect_from_name(p: Path):
    """
    파일명 규칙 (원본 주석 기준)
        neg_pos   : 위 neg, 아래 pos
        neg_half  : 위 neg, 아래 pos  (표적 DNA 와 타 DNA 반반)
        neg_other : 위 neg, 아래 neg  (타 DNA 만)
        pos_half  : 위 pos, 아래 pos
        *error*   : 검출 실패가 예상되는 이미지 → 평가에서 제외
    """
    s = p.stem.lower()
    if "error" in s:
        return None, None
    if "neg_other" in s:
        return "neg", "neg"
    if "neg_half" in s:
        return "neg", "pos"
    if "pos_half" in s:
        return "pos", "pos"
    if "neg_pos" in s:
        return "neg", "pos"
    if "pos_neg" in s:
        return "pos", "neg"
    return None, None


# ==================================================================
# 통계
# ==================================================================
def youden_threshold(values, labels):
    """
    민감도 + 특이도 - 1 이 최대가 되는 임계값.
    values >= T 이면 양성으로 판정한다고 가정한다.
    """
    v = np.asarray(values, float)
    y = np.asarray(labels, int)
    m = np.isfinite(v)
    v, y = v[m], y[m]
    if v.size < 4 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan, np.nan

    order = np.argsort(v)
    v, y = v[order], y[order]
    P_ = int((y == 1).sum())
    N_ = int((y == 0).sum())

    TP_right = np.cumsum((y == 1).astype(int)[::-1])[::-1]
    FP_right = np.cumsum((y == 0).astype(int)[::-1])[::-1]

    best = (-1.0, np.nan, np.nan, np.nan)   # J, T, TPR, FPR
    for i in np.where(np.diff(v) != 0)[0]:
        TPR = TP_right[i + 1] / P_
        FPR = FP_right[i + 1] / N_
        J = TPR - FPR
        if J > best[0]:
            best = (J, (v[i] + v[i + 1]) / 2.0, TPR, FPR)
    J, T, TPR, FPR = best
    return float(T), float(J), float(TPR), float(FPR)


def bootstrap_ci(func, *arrays, n=N_BOOTSTRAP, seed=RNG_SEED, alpha=0.05):
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
    """이항 비율의 Wilson 신뢰구간. 표본이 작을 때 정규근사보다 정확하다."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z**2 / n
    c = p + z**2 / (2 * n)
    s = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return {"point": p, "lo": float((c - s) / d), "hi": float((c + s) / d), "k": int(k), "n": int(n)}


# ==================================================================
# 메인
# ==================================================================
def main():
    ap = argparse.ArgumentParser(description="Step4: pair 비율 임계값 도출")
    ap.add_argument("--weights", default=str(P.WEIGHTS_PATH))
    ap.add_argument("--solo_train_root", default=str(P.SOLO_TRAIN))
    ap.add_argument("--solo_test_root", default=str(P.SOLO_TEST))
    ap.add_argument("--pair_negpos_root", default=str(P.PAIR_NEGPOS))
    ap.add_argument("--pair_negneg_root", default=str(P.PAIR_NEGNEG))
    ap.add_argument("--test_all_root", default=str(P.TEST_ALL))
    ap.add_argument("--out_dir", default=str(P.OUT_STEP4))
    ap.add_argument("--cutoff_abs", type=float, default=P.ABS_NEG_CUTOFF,
                    help="step3 에서 구한 절대 음성 기준선")
    ap.add_argument("--iou", type=float, default=P.IOU)
    ap.add_argument("--imgsz", type=int, default=P.IMG_SIZE)
    ap.add_argument("--device", default="")
    ap.add_argument("--save_viz", action="store_true")
    ap.add_argument("--use_threshold", choices=["negpos", "all", "youden"], default="all",
                    help=("test_all 평가에 쓸 임계값. "
                          "negpos=원본 조건(앱과 동일), all=neg_neg 포함, youden=직접 최적화"))
    args = ap.parse_args()

    CONF = P.CONF_MIN
    out_dir = P.ensure_dir(Path(args.out_dir))
    viz_dir = P.ensure_dir(out_dir / "viz") if args.save_viz else None

    P.check(Path(args.weights), Path(args.solo_train_root), Path(args.pair_negpos_root))

    print("=" * 64)
    print("Step4 · pair 비율 임계값 도출")
    print("=" * 64)
    print(f"  가중치     : {Path(args.weights).name}")
    print(f"  설정       : conf={CONF}, method=G, metric=p95")
    print(f"  step3 기준 : {args.cutoff_abs}")
    print(f"  출력       : {out_dir}")
    print(f"  오버레이   : {'저장' if args.save_viz else '생략'}")
    print()

    # ---------------- 모델 ----------------
    model = YOLO(str(args.weights))
    names = model.model.names if hasattr(model.model, "names") else model.names
    try:
        tube_cls = next(k for k, v in names.items() if str(v).lower() == "tube")
        roi_cls = next(k for k, v in names.items() if str(v).lower() == "roi")
    except StopIteration:
        raise SystemExit(f"tube/roi 클래스를 찾을 수 없습니다. names={names}")
    print(f"[MODEL] classes={names}  (tube={tube_cls}, roi={roi_cls})\n")

    def infer(path: Path):
        img = imread_unicode(path)
        if img is None:
            return None, [], []
        res = model.predict(source=img, imgsz=args.imgsz, conf=CONF,
                            iou=args.iou, device=args.device, verbose=False)[0]
        tubes, rois = [], []
        for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            (tubes if int(c) == tube_cls else rois).append(b)
        return img, tubes, rois

    # ================= 1. SOLO =================
    print("[1/4] solo 분석")
    solo_rows = []
    train_resolved = Path(args.solo_train_root).resolve()
    solo_imgs = list_images(args.solo_train_root) + list_images(args.solo_test_root)

    for i, ip in enumerate(solo_imgs, 1):
        label = solo_label_from_path(ip)
        if label is None:
            continue
        img, tubes, rois = infer(ip)
        if img is None or not rois:
            continue
        vals = [g_p95_intensity(safe_crop(img, r)) for r in rois]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            continue
        split = "train" if train_resolved in ip.resolve().parents else "test"
        solo_rows.append({"image_id": ip.stem, "image_path": str(ip),
                          "split": split, "label": label,
                          "I": f"{float(np.max(vals)):.6f}"})
        if viz_dir is not None:
            save_viz(img, tubes, rois, viz_dir / f"solo__{ip.stem}")
        if i % 30 == 0 or i == len(solo_imgs):
            print(f"    {i}/{len(solo_imgs)}")

    with open(out_dir / "solo_analysis.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "image_path", "split", "label", "I"])
        w.writeheader(); w.writerows(solo_rows)

    solo_neg = np.array([float(r["I"]) for r in solo_rows if r["label"] == "neg"])
    solo_pos = np.array([float(r["I"]) for r in solo_rows if r["label"] == "pos"])
    print(f"    solo neg={len(solo_neg)}, pos={len(solo_pos)}\n")

    # ================= 2. PAIR =================
    print("[2/4] pair 분석")
    pair_rows = []

    def scan_pair(root, group, lower_label):
        imgs = list_images(root)
        print(f"    {group}: {len(imgs)}장")
        for ip in imgs:
            img, tubes, rois = infer(ip)
            if img is None:
                pair_rows.append({"image_path": str(ip), "group": group,
                                  "lower_label": lower_label, "I_nc": "", "I_sample": "",
                                  "sat_nc": "", "sat_sample": "",
                                  "delta_pct": "", "ratio": "", "note": "IMREAD_FAIL"})
                continue
            rs = sorted(rois, key=center_y)
            if len(rs) < 2:
                pair_rows.append({"image_path": str(ip), "group": group,
                                  "lower_label": lower_label, "I_nc": "", "I_sample": "",
                                  "sat_nc": "", "sat_sample": "",
                                  "delta_pct": "", "ratio": "",
                                  "note": "ROI_PARTIAL" if len(rs) == 1 else "ROI_NONE"})
                if viz_dir is not None:
                    save_viz(img, tubes, rois, viz_dir / f"pair__{ip.stem}")
                continue
            cu, cl = safe_crop(img, rs[0]), safe_crop(img, rs[1])
            Iu, Il = g_p95_intensity(cu), g_p95_intensity(cl)
            su, sl = sat_frac(cu), sat_frac(cl)
            m = (Iu + Il) / 2.0
            delta = abs(Il - Iu) / m * 100.0 if m > 0 else np.nan
            ratio = Il / Iu if Iu > 0 else np.nan
            note = "SATURATED" if (np.isfinite(sl) and sl >= 0.05) else ""
            pair_rows.append({
                "image_path": str(ip), "group": group, "lower_label": lower_label,
                "I_nc": f"{Iu:.6f}", "I_sample": f"{Il:.6f}",
                "sat_nc": f"{su:.4f}", "sat_sample": f"{sl:.4f}",
                "delta_pct": f"{delta:.6f}" if np.isfinite(delta) else "",
                "ratio": f"{ratio:.6f}" if np.isfinite(ratio) else "",
                "note": note,
            })
            if viz_dir is not None:
                save_viz(img, tubes, [rs[0], rs[1]], viz_dir / f"pair__{ip.stem}")

    scan_pair(args.pair_negpos_root, "neg_pos", "pos")
    if Path(args.pair_negneg_root).exists():
        scan_pair(args.pair_negneg_root, "neg_neg", "neg")

    with open(out_dir / "pair_analysis.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "group", "lower_label",
                                          "I_nc", "I_sample", "sat_nc", "sat_sample",
                                          "delta_pct", "ratio", "note"])
        w.writeheader(); w.writerows(pair_rows)

    valid = [r for r in pair_rows if r["ratio"]]
    n_sat = sum(1 for r in valid if r["note"] == "SATURATED")
    if n_sat:
        print(f"    [주의] 시료 ROI 가 포화된 이미지 {n_sat}장. "
              f"포화되면 실제 형광 세기를 알 수 없어 ratio 가 과소평가된다.")
    nc_vals = np.array([float(r["I_nc"]) for r in valid])
    ratios = np.array([float(r["ratio"]) for r in valid])
    labels = np.array([1 if r["lower_label"] == "pos" else 0 for r in valid])
    print(f"    유효 pair {len(valid)}장 "
          f"(pos={int(labels.sum())}, neg={int((labels == 0).sum())})\n")

    if len(valid) == 0:
        raise SystemExit("유효한 pair 데이터가 없습니다.")

    # ================= 3. 임계값 도출 =================
    print("[3/4] 임계값 도출")

    def transfer_T(nc_array):
        med = float(np.median(nc_array))
        return (args.cutoff_abs / med if med > 0 else np.nan), med

    # --- [A-1] 원본 조건: neg_pos 만 ---
    nc_negpos = np.array([float(r["I_nc"]) for r in valid if r["group"] == "neg_pos"])
    T_a1, med_a1 = transfer_T(nc_negpos)
    ci_a1 = bootstrap_ci(lambda a: transfer_T(a)[0], nc_negpos)

    print(f"  [A-1] 전이 방식 · neg_pos 만  (2025-11-06 원본 조건)")
    print(f"        n={len(nc_negpos)}, median(NC)={med_a1:.2f}")
    print(f"        T = {args.cutoff_abs} / {med_a1:.2f} = {T_a1:.4f}")
    if ci_a1:
        print(f"        95% CI = [{ci_a1['lo']:.4f}, {ci_a1['hi']:.4f}]")

    # --- [A-2] 확장: neg_neg 포함 ---
    T_a2, med_a2 = transfer_T(nc_vals)
    ci_a2 = bootstrap_ci(lambda a: transfer_T(a)[0], nc_vals)

    print(f"  [A-2] 전이 방식 · neg_pos + neg_neg  (11-17 추가 데이터 포함)")
    print(f"        n={len(nc_vals)}, median(NC)={med_a2:.2f}")
    print(f"        T = {args.cutoff_abs} / {med_a2:.2f} = {T_a2:.4f}")
    if ci_a2:
        print(f"        95% CI = [{ci_a2['lo']:.4f}, {ci_a2['hi']:.4f}]")

    # --- [B] Youden's J ---
    T_youden = J = TPR = FPR = np.nan
    ci_youden = None
    if len(np.unique(labels)) == 2:
        T_youden, J, TPR, FPR = youden_threshold(ratios, labels)
        ci_youden = bootstrap_ci(lambda v, y: youden_threshold(v, y)[0], ratios, labels)
        print(f"  [B]   Youden's J · pair ratio 분포 직접 최적화")
        print(f"        T = {T_youden:.4f}  (J={J:.3f}, TPR={TPR:.3f}, FPR={FPR:.3f})")
        if ci_youden:
            print(f"        95% CI = [{ci_youden['lo']:.4f}, {ci_youden['hi']:.4f}]")
        print(f"        FPR {FPR*100:.1f}% 는 임계값을 어디에 두어도 이보다 낮출 수 없는")
        print(f"        최적점의 값이므로, 신호 분포 자체의 겹침을 뜻한다.")
    else:
        print("  [B]   Youden's J — 음성 pair 가 없어 계산 불가")

    T_map = {"negpos": T_a1, "all": T_a2, "youden": T_youden}
    T_used = T_map[args.use_threshold]
    med_used = med_a1 if args.use_threshold == "negpos" else med_a2

    # 세 값이 서로의 신뢰구간 안에 들어오는지
    all_T = [t for t in (T_a1, T_a2, T_youden) if np.isfinite(t)]
    if ci_a1 and len(all_T) > 1:
        within = all(ci_a1["lo"] <= t <= ci_a1["hi"] for t in all_T)
        print(f"\n  세 임계값 범위: {min(all_T):.4f} ~ {max(all_T):.4f}")
        print(f"  {'모두 A-1 의 신뢰구간 안에 있어 통계적으로 구분되지 않는다.' if within else '일부가 A-1 의 신뢰구간을 벗어난다.'}")

    print(f"\n  → test_all 평가에 사용할 임계값: {T_used:.4f} ({args.use_threshold})\n")

    # 판정 마진과 NC 변동성 비교
    nc_cv = float(np.std(nc_vals, ddof=1) / np.mean(nc_vals) * 100) if len(nc_vals) > 1 else np.nan
    margin_pct = (T_used - 1.0) * 100

    threshold_json = {
        "step3_cutoff_abs": args.cutoff_abs,
        "pair_nc": {
            "n_all": len(nc_vals), "median_all": med_a2,
            "n_negpos": len(nc_negpos), "median_negpos": med_a1,
            "mean": float(np.mean(nc_vals)), "sd": float(np.std(nc_vals, ddof=1)),
            "cv_pct": nc_cv,
        },
        "method_A1_transfer_negpos_only": {
            "description": ("step3 절대 cutoff 를 neg_pos pair 의 NC 중앙값으로 나눈 값. "
                            "2025-11-06 원본 도출 조건이며 앱의 RATIO_THR 과 일치한다."),
            "T_ratio": T_a1, "n_nc": len(nc_negpos), "median_nc": med_a1,
            "bootstrap_ci_95": ci_a1,
        },
        "method_A2_transfer_all_pairs": {
            "description": ("11-17 에 추가 촬영한 neg_neg pair 의 NC 까지 포함해 재계산. "
                            "NC 표본이 늘어 중앙값이 더 안정적이다."),
            "T_ratio": T_a2, "n_nc": len(nc_vals), "median_nc": med_a2,
            "bootstrap_ci_95": ci_a2,
        },
        "method_B_youden": {
            "description": "pair ratio 분포에서 민감도+특이도 최대 지점",
            "T_ratio": T_youden if np.isfinite(T_youden) else None,
            "youden_J": J if np.isfinite(J) else None,
            "TPR": TPR if np.isfinite(TPR) else None,
            "FPR": FPR if np.isfinite(FPR) else None,
            "bootstrap_ci_95": ci_youden,
            "n_pos": int(labels.sum()), "n_neg": int((labels == 0).sum()),
            "note": ("최적점에서도 FPR 이 0 이 아니라면, 임계값 조정으로는 "
                     "위양성을 더 줄일 수 없다는 뜻이다."),
        },
        "used_for_test_all": args.use_threshold,
        "T_used": T_used,
        "robustness_note": {
            "판정_마진_pct": margin_pct,
            "pair_NC_변동계수_pct": nc_cv,
            "해석": ("판정 마진이 NC 변동계수와 비슷하거나 작으면, "
                     "기기·조명 차이만으로 판정이 뒤집힐 수 있다."),
        },
    }
    (out_dir / "threshold_derivation.json").write_text(
        json.dumps(threshold_json, indent=2, ensure_ascii=False), encoding="utf-8")

    # pair 자체 성능 (참고용, 임계값을 여기서 골랐으므로 낙관적일 수 있음)
    if len(np.unique(labels)) == 2:
        pred = (ratios >= T_used).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        tn = int(((pred == 0) & (labels == 0)).sum())
        print(f"  pair 자체 성능: TP={tp} FN={fn} FP={fp} TN={tn}")
        acc_ci = wilson_ci(tp + tn, len(labels))
        if acc_ci:
            print(f"    정확도 {acc_ci['point']*100:.1f}% "
                  f"(95% CI {acc_ci['lo']*100:.1f}–{acc_ci['hi']*100:.1f}%)")
        print()

    # ================= 4. TEST_ALL =================
    print("[4/4] test_all 최종 검증")
    print("  주의: 이 단계는 평가 전용이다. 여기 결과로 임계값을 바꾸지 않는다.\n")

    test_rows = []
    for dev_name, dev_root in P.DEVICE_SETS.items():
        if not Path(dev_root).exists():
            continue
        imgs = list_images(dev_root)
        print(f"  {dev_name}: {len(imgs)}장")
        for ip in imgs:
            ue, le = expect_from_name(ip)
            img, tubes, rois = infer(ip)

            base = {"device": dev_name, "image": ip.name,
                    "upper_exp": ue or "", "lower_exp": le or ""}

            if img is None or len(rois) < 2:
                test_rows.append({**base, "Iu": "", "Il": "", "ratio": "",
                                  "upper_pred": "", "lower_pred": "",
                                  "note": "ROI_NONE" if not rois else "ROI_PARTIAL"})
                continue

            rs = sorted(rois, key=center_y)
            Iu = g_p95_intensity(safe_crop(img, rs[0]))
            Il = g_p95_intensity(safe_crop(img, rs[1]))
            ratio = Il / Iu if Iu > 0 else np.nan

            # 위쪽: 비교 대상이 없으므로 절대 기준으로 판정
            upper_pred = "pos" if (np.isfinite(Iu) and Iu >= args.cutoff_abs) else "neg"
            # 아래쪽: NC 대비 비율로 판정 (앱과 동일)
            lower_pred = "pos" if (np.isfinite(ratio) and ratio >= T_used) else "neg"

            test_rows.append({**base,
                              "Iu": f"{Iu:.4f}", "Il": f"{Il:.4f}",
                              "ratio": f"{ratio:.4f}" if np.isfinite(ratio) else "",
                              "upper_pred": upper_pred, "lower_pred": lower_pred,
                              "note": ""})
            if viz_dir is not None:
                save_viz(img, tubes, [rs[0], rs[1]], viz_dir / f"test__{dev_name}__{ip.stem}")

    with open(out_dir / "test_all_eval.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["device", "image", "upper_exp", "lower_exp",
                                          "Iu", "Il", "ratio",
                                          "upper_pred", "lower_pred", "note"])
        w.writeheader(); w.writerows(test_rows)

    # 정확도 집계 (기대값이 있고 검출에 성공한 것만)
    ev = [r for r in test_rows if r["upper_exp"] and r["lower_exp"] and not r["note"]]
    n_skip = len(test_rows) - len(ev)

    def acc_of(key_exp, key_pred):
        k = sum(1 for r in ev if r[key_exp] == r[key_pred])
        return wilson_ci(k, len(ev))

    up_ci = acc_of("upper_exp", "upper_pred")
    lo_ci = acc_of("lower_exp", "lower_pred")
    both_k = sum(1 for r in ev
                 if r["upper_exp"] == r["upper_pred"] and r["lower_exp"] == r["lower_pred"])
    both_ci = wilson_ci(both_k, len(ev))

    print()
    print("=" * 64)
    print(f"  평가 대상 {len(ev)}장 (제외 {n_skip}장: 검출 실패 또는 라벨 없음)")
    if lo_ci:
        print(f"  아래(시료) 정확도 : {lo_ci['point']*100:5.1f}%  "
              f"({lo_ci['k']}/{lo_ci['n']})  95% CI {lo_ci['lo']*100:.1f}–{lo_ci['hi']*100:.1f}%")
    if up_ci:
        print(f"  위(NC)   정확도 : {up_ci['point']*100:5.1f}%  "
              f"({up_ci['k']}/{up_ci['n']})  95% CI {up_ci['lo']*100:.1f}–{up_ci['hi']*100:.1f}%")
        if all(r["upper_exp"] == "neg" for r in ev):
            print("      ※ 평가 대상에 위쪽이 양성인 이미지가 없다.")
            print("        위쪽을 항상 음성이라고 답해도 같은 값이 나오므로,")
            print("        이 수치는 위쪽 판정 능력을 검증하지 못한다.")
    if both_ci:
        print(f"  둘 다 맞은 비율   : {both_ci['point']*100:5.1f}%  "
              f"({both_ci['k']}/{both_ci['n']})  95% CI {both_ci['lo']*100:.1f}–{both_ci['hi']*100:.1f}%")

    # 기기별
    print("-" * 64)
    for dev in P.DEVICE_SETS:
        sub = [r for r in ev if r["device"] == dev]
        if not sub:
            continue
        k = sum(1 for r in sub if r["lower_exp"] == r["lower_pred"])
        nc_list = [float(r["Iu"]) for r in sub if r["Iu"]]
        nc_txt = f"NC 평균 {np.mean(nc_list):6.1f}" if nc_list else ""
        print(f"  {dev:14s} 아래 정확도 {k}/{len(sub)}   {nc_txt}")
    print("=" * 64)

    summary = {
        "settings": {"conf": CONF, "method": "G", "metric": "p95",
                     "iou": args.iou, "imgsz": args.imgsz},
        "threshold": threshold_json,
        "test_all": {
            "n_evaluated": len(ev), "n_skipped": n_skip,
            "lower_accuracy": lo_ci, "upper_accuracy": up_ci, "both_accuracy": both_ci,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[저장] {out_dir}")


if __name__ == "__main__":
    main()