# step4_ratio_threshold.py
# 이전 step에서 얻었던 결과 활용: conf=0.70, method='G', metric='p95' 고정할 것 
# step 3 결과 negative와 positive를 가르는 negative cutoff는 221.000였음 
# JSON 불러오지 않고 모델로 직접 ROI 검출 → 밝기 계산(G-p95) 할 것 # step 3에서는 Solo 데이터만 사용했다면, step4에서는 pair 데이터 (이미지의 roi 박스가 두개 검출되고 그 중 위는 negative, 아래는 positive)를 사용 
# 앱 구동 시에는 사용자가 nc와 자신의 시료를 같이 촬영함. 해당 nc의 형광값을 분석하여 자신의 시료 형광값을 보정하고 싶어서 step4를 진행하는 것임 
# 어떻게 보정할 것인지에 관한 코드를 작성하고 분석 결과를 도출하는 것이 목표! 
# 너가 이전에 pair의 Δ%을 계산하고, Youden's J를 해야 한다고 했는데, 하는 게 맞을지 한다면 어떻게 해야할지 감이 안잡힘 (지금 사진에는 위는 무조건 neg, 아래는 무조건 pos밖에 없음) 
# pair의 nc와 단일 negative 간 형광값 차이 확인 (이것도 평균으로 봐야할지, 상위 몇개로만 봐야할지, 하위 몇개로만 봐야할지 고민중임) 
# pair의 positive과 단일 positive 간 형광값 차이 확인 (필요없다고 생각하면 안해도 됨) 
# 마지막에 test_all 폴더 사용: 최종 테스트 및 결과 확인용 
# test_all 폴더에는 다음과 같은 하위 폴더 존재: "C:\n.gonorrhea_diagnostic_app\dataset\test_all\neg_pos_iphone13pro", "C:\n.gonorrhea_diagnostic_app\dataset\test_all\neg_pos_galaxynote8", "C:\n.gonorrhea_diagnostic_app\dataset\test_all\neg_pos_iphone13" 
# galaxynote8하위 폴더에는 다음과 같은 이미지 이름 존재: galaxy_neg_half (1장), galaxy_neg_other (1장) ,galaxy_neg_pos (5장), galaxt_neg_pos_error (2장) 
# 다른 하위 폴더에도 유사하게 존재. 
# test_all의 이미지에서 나와야 하는 이상적인 결과: 
# neg_pos: 두 튜브 중 상위 튜브는 negative, 하나는 positive로 잡혀야 함 
# neg_half: 두 튜브 중 상위 튜브는 negative, 하나는 positive로 잡혀야 함 (template DNA와 타 DNA가 반반 들어갔음) 
# neg_other: 두 튜브 중 상위 튜브는 negative, 하위 튜브는 negative로 잡혀야 함 (타 DNA가 들어갔음) 
# pos_half: 두 튜브 중 상위 튜브는 positive, 하위 튜브는 positive로 잡혀야 함 (나중에 얍 구동 시 상위 튜브는 무조건 nc만 오게 할 예정. 일단 positive, negative를 잘 잡는지 확인하는 것이 step4의 목적)
# step4_pair_solo_relation.py  (revised full)

import argparse, os, csv, json, re
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Ultralytics YOLO가 필요합니다. pip install ultralytics") from e

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def log(msg: str):
    print(str(msg), flush=True)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def list_images(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]

def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def safe_crop(img, xyxy):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    H, W = img.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W - 1, x2); y2 = min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def center_y(b):
    return (float(b[1]) + float(b[3])) / 2.0

def solo_label_from_path(p: Path) -> str | None:
    low = str(p).lower()
    if re.search(r"[/\\]pos[/\\]", low):
        return "pos"
    if re.search(r"[/\\]neg[/\\]", low):
        return "neg"
    return None

def expect_from_test_all_name(p: Path):
    s = p.stem.lower()
    upper_exp = lower_exp = ""
    if "neg_pos" in s or "pos_neg" in s:
        upper_exp, lower_exp = ("neg", "pos") if "neg_pos" in s else ("pos", "neg")
    elif "neg_other" in s:
        upper_exp, lower_exp = ("neg", "neg")
    elif "neg_half" in s:
        upper_exp, lower_exp = ("neg", "pos")
    elif "pos_half" in s or re.search(r"_pos(?:_|$)", s):
        upper_exp, lower_exp = ("pos", "pos")
    if "error" in s:
        lower_exp = "error"
    return upper_exp, lower_exp

def g_p95_intensity(crop_bgr: np.ndarray) -> float:
    if crop_bgr is None:
        return np.nan
    g = crop_bgr[:, :, 1].astype(np.float32)
    flat = g.reshape(-1)
    if flat.size == 0:
        return np.nan
    return float(np.percentile(flat, 95))

def draw_boxes(image, tubes, rois, save_path):
    draw = image.copy()
    for xyxy in tubes:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for xyxy in rois:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(draw, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.imwrite(str(save_path), draw)

def apply_calib_values(I_nc, I_sm, mode, k, b):
    if mode == "ratio":
        return k * I_nc, k * I_sm
    elif mode == "shift":
        return I_nc + b, I_sm + b
    elif mode == "affine":
        return k * I_nc + b, k * I_sm + b
    else:
        return I_nc, I_sm

def fit_affine_from_two_points(ref_neg, cutoff_abs, ref_pos, target_pos):
    if not (np.isfinite(ref_neg) and np.isfinite(ref_pos) and (ref_pos - ref_neg) != 0):
        return np.nan, np.nan
    k = (target_pos - cutoff_abs) / (ref_pos - ref_neg)
    b = cutoff_abs - k * ref_neg
    return float(k), float(b)

def best_cutoff_youden(values: np.ndarray, labels: np.ndarray):
    mask = np.isfinite(values) & np.isfinite(labels)
    v = values[mask]; y = labels[mask].astype(int)
    if v.size < 4 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan, np.nan
    order = np.argsort(v)
    v_sorted = v[order]; y_sorted = y[order]
    P = (y_sorted == 1).sum(); N = (y_sorted == 0).sum()
    if P == 0 or N == 0:
        return np.nan, np.nan, np.nan, np.nan
    counts_pos = (y_sorted == 1).astype(int)
    counts_neg = (y_sorted == 0).astype(int)
    TP_right = np.cumsum(counts_pos[::-1])[::-1]
    FP_right = np.cumsum(counts_neg[::-1])[::-1]
    best_J = -1.0; best_T = np.nan; best_TPR = np.nan; best_FPR = np.nan
    uniq_idx = np.where(np.diff(v_sorted) != 0)[0]
    for i in uniq_idx:
        TP = TP_right[i + 1]; FP = FP_right[i + 1]
        TPR = TP / P; FPR = FP / N
        J = TPR - FPR
        if (J > best_J) or (np.isclose(J, best_J) and TPR > best_TPR):
            best_J = J; best_TPR = TPR; best_FPR = FPR
            best_T = (v_sorted[i] + v_sorted[i + 1]) / 2.0
    return float(best_T), float(best_J), float(best_TPR), float(best_FPR)

def main():
    ap = argparse.ArgumentParser()
    # 경로
    ap.add_argument("--weights", required=True)
    ap.add_argument("--solo_train_root", default=r"C:\n.gonorrhea_diagnostic_app\dataset\solo\train")
    ap.add_argument("--solo_test_root",  default=r"C:\n.gonorrhea_diagnostic_app\dataset\solo\test")
    ap.add_argument("--pair_roots", nargs="+", default=[r"C:\n.gonorrhea_diagnostic_app\dataset\pair\neg_pos"])
    ap.add_argument("--test_all_roots", nargs="*", default=[
        r"C:\n.gonorrhea_diagnostic_app\dataset\test_all\neg_pos_iphone13pro",
        r"C:\n.gonorrhea_diagnostic_app\dataset\test_all\neg_pos_galaxynote8",
        r"C:\n.gonorrhea_diagnostic_app\dataset\test_all\neg_pos_iphone13",
    ])
    ap.add_argument("--out_dir", default=r"C:\n.gonorrhea_diagnostic_app\analysis_output\step4")
    # 모델/추론
    ap.add_argument("--conf", type=float, default=0.70)
    ap.add_argument("--iou",  type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--save_viz", action="store_true")
    # 임계값(기본값): Δ%/ratio 자동
    ap.add_argument("--t_delta", type=float, default=float("nan"))
    ap.add_argument("--t_ratio", type=float, default=float("nan"))
    # 보정
    ap.add_argument("--calib", type=str, default="none",
                    choices=["none", "ratio", "shift", "affine"])
    ap.add_argument("--calib_auto", type=str, default="on",
                    choices=["off", "on"])
    # pair 전용 절대 cut-off 사용 여부
    ap.add_argument("--use_pair_abs", type=str, default="on",
                    choices=["off", "on"])
    # step3 고정 cutoff (neg 절대기준)
    ap.add_argument("--cutoff_abs", type=float, default=221.0)
    # 빠른 점검: 앞 N장만
    ap.add_argument("--max_per_set", type=int, default=0)
    # 시나리오 비교 프리셋
    ap.add_argument("--compare_presets", type=str, default="off", choices=["off", "on"])
    ap.add_argument("--t_delta_list", type=str, default="5,7.5,10", help="B/D에서 사용할 Δ% 목록. 예: 5,7.5,10")

    args = ap.parse_args()
    out_dir = ensure_dir(Path(args.out_dir))
    viz_dir = ensure_dir(out_dir / "viz") if args.save_viz else None

    log("=== Step4: pair-solo relation / ratiometric + optional absolute/Δ% ===")
    log(f"[CONFIG] conf={args.conf} iou={args.iou} imgsz={args.imgsz} method=G metric=p95 cutoff_abs={args.cutoff_abs}")
    log(f"[CONFIG] calib_auto={args.calib_auto} use_pair_abs={args.use_pair_abs} device='{args.device or 'auto'}'")

    # ----- YOLO 로드 -----
    log("[MODEL] Loading YOLO weights...")
    model = YOLO(args.weights)
    names = model.model.names if hasattr(model.model, "names") else model.names
    try:
        tube_cls = [k for k, v in names.items() if str(v).lower() == "tube"][0]
        roi_cls  = [k for k, v in names.items() if str(v).lower() == "roi"][0]
    except Exception:
        raise RuntimeError(f"클래스 이름을 찾지 못했습니다. names={names}")
    log(f"[MODEL] Classes: tube={tube_cls}, roi={roi_cls}")

    def infer_one(path: Path):
        res = model.predict(source=str(path), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                            device=args.device, verbose=False)[0]
        img = cv2.imread(str(path))
        if img is None:
            return None, [], []
        tubes, rois = [], []
        for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy()):
            if int(c) == tube_cls:
                tubes.append(b)
            elif int(c) == roi_cls:
                rois.append(b)
        return img, tubes, rois

    # ================= SOLO =================
    solo_rows = []
    train_root, test_root = Path(args.solo_train_root), Path(args.solo_test_root)
    solo_imgs = list_images(train_root) + list_images(test_root)
    if args.max_per_set > 0:
        solo_imgs = solo_imgs[:args.max_per_set]
    log(f"[SOLO] images: {len(solo_imgs)}")
    for i, ip in enumerate(solo_imgs, 1):
        if (i == 1) or (i % 10 == 0) or (i == len(solo_imgs)):
            log(f"[SOLO] {i}/{len(solo_imgs)}: {ip.name}")
        label = solo_label_from_path(ip)
        if label not in {"pos", "neg"}:
            continue
        split = "train" if str(train_root).lower() in str(ip).lower() else ("test" if str(test_root).lower() in str(ip).lower() else None)
        if split is None:
            continue
        img, tubes, rois = infer_one(ip)
        if img is None:
            continue
        vals = []
        for r in rois:
            crop = safe_crop(img, r)
            vals.append(g_p95_intensity(crop))
        if len(vals) == 0:
            continue
        I = float(np.max(vals))
        solo_rows.append({"image_id": ip.stem, "image_path": str(ip), "split": split, "label": label, "I_raw": f"{I:.6f}"})
        if viz_dir is not None:
            draw_boxes(img, tubes, rois, viz_dir / f"solo__{ip.stem}.jpg")

    solo_csv = out_dir / "solo_analysis.csv"
    with open(solo_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "image_path", "split", "label", "I_raw"])
        w.writeheader()
        for r in solo_rows:
            w.writerow(r)

    solo_neg_vals = [to_float(r["I_raw"]) for r in solo_rows if r["label"] == "neg"]
    solo_pos_vals = [to_float(r["I_raw"]) for r in solo_rows if r["label"] == "pos"]
    solo_train_neg = [to_float(r["I_raw"]) for r in solo_rows if r["split"] == "train" and r["label"] == "neg"]
    mean_solo_neg   = float(np.mean(solo_train_neg)) if len(solo_train_neg) > 0 else np.nan
    median_solo_neg = float(np.median(solo_neg_vals)) if len(solo_neg_vals) > 0 else np.nan
    median_solo_pos = float(np.median(solo_pos_vals)) if len(solo_pos_vals) > 0 else np.nan
    cutoff_abs_solo = float(args.cutoff_abs)

    # ================= PAIR =================
    pair_rows = []
    pair_imgs = []
    for r in args.pair_roots:
        pair_imgs += list_images(Path(r))
    if args.max_per_set > 0:
        pair_imgs = pair_imgs[:args.max_per_set]
    log(f"[PAIR] images: {len(pair_imgs)}")

    I_nc_list = []; I_sm_list = []
    for i, ip in enumerate(pair_imgs, 1):
        if (i == 1) or (i % 5 == 0) or (i == len(pair_imgs)):
            log(f"[PAIR] {i}/{len(pair_imgs)}: {ip.name}")
        img, tubes, rois = infer_one(ip)
        if img is None:
            pair_rows.append({"image_path": str(ip), "I_nc": "", "I_sample": "", "delta_pct": "", "ratio": "",
                              "rule": "", "pred_upper": "", "pred_lower": "", "note": "IMREAD_FAIL"})
            continue
        rois_sorted = sorted(rois, key=center_y)
        if len(rois_sorted) < 2:
            note = "ROI_PARTIAL" if len(rois_sorted) == 1 else "ROI_NONE"
            pair_rows.append({"image_path": str(ip), "I_nc": "", "I_sample": "", "delta_pct": "", "ratio": "",
                              "rule": "", "pred_upper": "", "pred_lower": "", "note": note})
            if viz_dir is not None:
                draw_boxes(img, tubes, rois, viz_dir / f"pair__{ip.stem}.jpg")
            continue

        ur, lr = rois_sorted[0], rois_sorted[1]
        Iu = g_p95_intensity(safe_crop(img, ur))
        Il = g_p95_intensity(safe_crop(img, lr))
        m = (Iu + Il) / 2.0
        delta = abs(Il - Iu) / m * 100.0 if m > 0 else np.nan
        ratio = (Il / Iu) if (Iu > 0) else np.nan
        I_nc_list.append(Iu); I_sm_list.append(Il)

        pair_rows.append({"image_path": str(ip), "I_nc": f"{Iu:.6f}", "I_sample": f"{Il:.6f}",
                          "delta_pct": f"{delta:.6f}" if np.isfinite(delta) else "",
                          "ratio": f"{ratio:.6f}" if np.isfinite(ratio) else "",
                          "rule": "", "pred_upper": "", "pred_lower": "", "note": ""})
        if viz_dir is not None:
            draw_boxes(img, tubes, [ur, lr], viz_dir / f"pair__{ip.stem}.jpg")

    pair_csv = out_dir / "pair_analysis.csv"
    with open(pair_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "I_nc", "I_sample", "delta_pct", "ratio",
                                          "rule", "pred_upper", "pred_lower", "note"])
        w.writeheader()
        for r in pair_rows:
            w.writerow(r)

    median_pair_nc = float(np.median(I_nc_list)) if len(I_nc_list) > 0 else np.nan
    median_pair_sm = float(np.median(I_sm_list)) if len(I_sm_list) > 0 else np.nan

    def auto_ratio_thr(ref_nc_median):
        if not (np.isfinite(cutoff_abs_solo) and np.isfinite(ref_nc_median) and ref_nc_median > 0):
            return np.nan
        return float(cutoff_abs_solo / ref_nc_median)

    # ---------- test_all 평가 함수 ----------
    def eval_on_test_all(T_delta, T_ratio, calib_desc, k=np.nan, b=np.nan, T_abs_pair=np.nan, use_pair_abs=False):
        rows = []
        il_values = []; il_labels = []

        for root in args.test_all_roots:
            test_list = list_images(Path(root))
            if args.max_per_set > 0:
                test_list = test_list[:args.max_per_set]
            log(f"[TEST_ALL] {root}: {len(test_list)}")

            for i, ip in enumerate(test_list, 1):
                if (i == 1) or (i % 5 == 0) or (i == len(test_list)):
                    log(f"[TEST_ALL] {Path(root).name} {i}/{len(test_list)}: {ip.name}")
                img, tubes, rois = infer_one(ip)

                if img is None:
                    # Iu/Il 자리 포함 (빈칸)
                    rows.append([str(ip), "", "", "", "", "", "",
                                 "", "", "IMREAD_FAIL",
                                 calib_desc, k, b, T_delta, T_ratio, T_abs_pair, use_pair_abs])
                    continue

                rois_sorted = sorted(rois, key=center_y)
                ue, le = expect_from_test_all_name(ip)

                if len(rois_sorted) < 2:
                    upper_pred = "neg" if len(rois_sorted) >= 1 else ""
                    lower_pred = "error" if len(rois_sorted) == 1 else ""
                    rows.append([str(ip), ue, le, upper_pred, lower_pred,
                                 "", "",  # Iu, Il
                                 "", "", "ROI_PARTIAL" if len(rois_sorted) == 1 else "ROI_NONE",
                                 calib_desc, k, b, T_delta, T_ratio, T_abs_pair, use_pair_abs])
                    continue

                ur, lr = rois_sorted[0], rois_sorted[1]
                Iu = g_p95_intensity(safe_crop(img, ur))
                Il = g_p95_intensity(safe_crop(img, lr))
                Iu_c, Il_c = apply_calib_values(Iu, Il, calib_desc.split("|")[0], k, b)

                m = (Iu_c + Il_c) / 2.0
                delta = abs(Il_c - Iu_c) / m * 100.0 if m > 0 else np.nan
                ratio = (Il_c / Iu_c) if (Iu_c > 0) else np.nan

                pos_by_delta = (np.isfinite(T_delta) and np.isfinite(delta) and delta >= T_delta)
                pos_by_ratio = (np.isfinite(T_ratio) and np.isfinite(ratio) and ratio >= T_ratio)
                pos_by_abs   = (use_pair_abs and np.isfinite(T_abs_pair) and np.isfinite(Il_c) and (Il_c >= T_abs_pair))
                is_pos = (pos_by_delta or pos_by_ratio or pos_by_abs)

                up, lp = ("neg", "pos") if is_pos else ("neg", "neg")

                rows.append([str(ip), ue, le, up, lp,
                             f"{Iu_c:.6f}" if np.isfinite(Iu_c) else "",
                             f"{Il_c:.6f}" if np.isfinite(Il_c) else "",
                             f"{delta:.6f}" if np.isfinite(delta) else "",
                             f"{ratio:.6f}" if np.isfinite(ratio) else "",
                             "", calib_desc, k, b, T_delta, T_ratio, T_abs_pair, use_pair_abs])

                if le in ("pos", "neg") and np.isfinite(Il_c):
                    il_values.append(Il_c); il_labels.append(1 if le == "pos" else 0)

        df = pd.DataFrame(rows, columns=[
            "image_path", "upper_exp", "lower_exp", "upper_pred", "lower_pred",
            "Iu", "Il", "delta_pct", "ratio", "note",
            "calib", "k", "b", "T_delta", "T_ratio", "T_abs_pair", "use_pair_abs"
        ])

        eval_df = df[(df["upper_exp"] != "") & (df["lower_exp"] != "")]
        upper_acc = (eval_df["upper_exp"] == eval_df["upper_pred"]).mean() if eval_df.shape[0] > 0 else np.nan
        lower_acc = (eval_df["lower_exp"] == eval_df["lower_pred"]).mean() if eval_df.shape[0] > 0 else np.nan
        both_acc  = ((eval_df["upper_exp"] == eval_df["upper_pred"]) &
                     (eval_df["lower_exp"] == eval_df["lower_pred"])).mean() if eval_df.shape[0] > 0 else np.nan

        valid_df = eval_df[(eval_df["note"] == "") | (eval_df["note"].isna())]
        f_upper = (valid_df["upper_exp"] == valid_df["upper_pred"]).mean() if valid_df.shape[0] > 0 else np.nan
        f_lower = (valid_df["lower_exp"] == valid_df["lower_pred"]).mean() if valid_df.shape[0] > 0 else np.nan
        f_both  = ((valid_df["upper_exp"] == valid_df["upper_pred"]) &
                   (valid_df["lower_exp"] == valid_df["lower_pred"])).mean() if valid_df.shape[0] > 0 else np.nan

        pair_abs_fit = None
        if len(il_values) >= 4 and (len(set(il_labels)) == 2):
            vals = np.array(il_values, dtype=float)
            labs = np.array(il_labels, dtype=int)
            T_abs, J, TPR, FPR = best_cutoff_youden(vals, labs)
            pair_abs_fit = {
                "T_abs_pair": float(T_abs) if np.isfinite(T_abs) else None,
                "YoudenJ": float(J) if np.isfinite(J) else None,
                "TPR": float(TPR) if np.isfinite(TPR) else None,
                "FPR": float(FPR) if np.isfinite(FPR) else None
            }

        return (upper_acc, lower_acc, both_acc, f_upper, f_lower, f_both, len(eval_df), len(valid_df)), rows, pair_abs_fit

    # ---------- 기준 후보 ----------
    ref_candidates = []
    if np.isfinite(mean_solo_neg):
        ref_candidates.append(("ref_neg=soloNegMean", "neg", mean_solo_neg))
    if np.isfinite(median_solo_neg):
        ref_candidates.append(("ref_neg=soloNegMedian", "neg", median_solo_neg))
    if np.isfinite(median_pair_nc):
        ref_candidates.append(("ref_neg=pairNCMedian", "neg", median_pair_nc))
    if np.isfinite(median_solo_neg) and np.isfinite(median_pair_nc):
        ref_candidates.append(("ref_neg=blendMedian", "neg", float(np.median([median_solo_neg, median_pair_nc]))))

    ref_pos = median_solo_pos if np.isfinite(median_solo_pos) else np.nan
    R_sp = (median_solo_pos / median_solo_neg) if (np.isfinite(median_solo_pos) and np.isfinite(median_solo_neg) and median_solo_neg > 0) else np.nan
    target_pos = (cutoff_abs_solo * R_sp) if (np.isfinite(R_sp) and R_sp > 1.0) else (cutoff_abs_solo * 1.2 if np.isfinite(cutoff_abs_solo) else np.nan)

    # ---------- 스윕 + 선택 ----------
    def run_one_setup(tag: str, force_t_delta, force_use_pair_abs: bool):
        T_delta_base = force_t_delta
        T_ratio_base = args.t_ratio if np.isfinite(args.t_ratio) else auto_ratio_thr(median_pair_nc)

        sweep_rows = []
        best = {"both_acc": -1, "lower_acc": -1, "upper_acc": -1, "setup": None, "rows": None, "pair_abs": None,
                "f_upper": -1, "f_lower": -1, "f_both": -1, "n_raw": 0, "n_filtered": 0}

        calib_modes = ["none", "ratio", "shift", "affine"] if args.calib_auto == "on" else [args.calib]

        # base(보정 없음)
        k = b = np.nan
        T_delta = T_delta_base
        T_ratio = T_ratio_base
        acc, rows, pair_abs_fit = eval_on_test_all(T_delta, T_ratio, "none|base", k, b, np.nan, use_pair_abs=False)
        if acc:
            ua, la, ba, fua, fla, fba, n_raw, n_f = acc
            sweep_rows.append(["none", "base", "", "", T_delta, T_ratio, False, np.nan, ua, la, ba, fua, fla, fba, n_raw, n_f,
                               None if pair_abs_fit is None else pair_abs_fit.get("T_abs_pair"),
                               None if pair_abs_fit is None else pair_abs_fit.get("YoudenJ")])
            if (ba > best["both_acc"]) or (ba == best["both_acc"] and la > best["lower_acc"]):
                best.update({"both_acc": ba, "lower_acc": la, "upper_acc": ua,
                             "f_upper": fua, "f_lower": fla, "f_both": fba,
                             "n_raw": n_raw, "n_filtered": n_f,
                             "setup": ("none", "base", k, b, T_delta, T_ratio, np.nan, False),
                             "rows": rows, "pair_abs": pair_abs_fit})

        for mode in calib_modes:
            if mode == "none":
                continue

            if mode in ("ratio", "shift"):
                for tagref, kind, ref_neg in ref_candidates:
                    if not (np.isfinite(ref_neg) and np.isfinite(cutoff_abs_solo)):
                        continue

                    if mode == "ratio":
                        k = float(cutoff_abs_solo / ref_neg) if ref_neg > 0 else np.nan
                        b = 0.0
                    else:
                        k = 1.0
                        b = float(cutoff_abs_solo - ref_neg)

                    ref_nc_median_adj = float(np.median([apply_calib_values(v, v, mode, k, b)[0] for v in I_nc_list])) if len(I_nc_list) > 0 else np.nan
                    T_delta = T_delta_base
                    T_ratio = args.t_ratio if np.isfinite(args.t_ratio) else \
                              (float(cutoff_abs_solo / ref_nc_median_adj) if (np.isfinite(ref_nc_median_adj) and ref_nc_median_adj > 0) else np.nan)

                    acc, rows, pair_abs_fit = eval_on_test_all(T_delta, T_ratio, f"{mode}|{tagref}", k, b, np.nan, use_pair_abs=False)
                    if acc:
                        ua, la, ba, fua, fla, fba, n_raw, n_f = acc
                        sweep_rows.append([mode, tagref, f"{k:.6f}", f"{b:.6f}", T_delta, T_ratio, False, np.nan,
                                           ua, la, ba, fua, fla, fba, n_raw, n_f,
                                           None if pair_abs_fit is None else pair_abs_fit.get("T_abs_pair"),
                                           None if pair_abs_fit is None else pair_abs_fit.get("YoudenJ")])
                        if (ba > best["both_acc"]) or (ba == best["both_acc"] and la > best["lower_acc"]):
                            best.update({"both_acc": ba, "lower_acc": la, "upper_acc": ua,
                                         "f_upper": fua, "f_lower": fla, "f_both": fba,
                                         "n_raw": n_raw, "n_filtered": n_f,
                                         "setup": (mode, tagref, k, b, T_delta, T_ratio, np.nan, False),
                                         "rows": rows, "pair_abs": pair_abs_fit})

                    if force_use_pair_abs and pair_abs_fit and pair_abs_fit.get("T_abs_pair") is not None:
                        T_abs_pair = float(pair_abs_fit["T_abs_pair"])
                        acc2, rows2, _ = eval_on_test_all(T_delta, T_ratio, f"{mode}|{tagref}", k, b, T_abs_pair, use_pair_abs=True)
                        if acc2:
                            ua2, la2, ba2, fua2, fla2, fba2, n_raw2, n_f2 = acc2
                            sweep_rows.append([mode, tagref, f"{k:.6f}", f"{b:.6f}", T_delta, T_ratio, True, T_abs_pair,
                                               ua2, la2, ba2, fua2, fla2, fba2, n_raw2, n_f2,
                                               pair_abs_fit.get("T_abs_pair"), pair_abs_fit.get("YoudenJ")])
                            if (ba2 > best["both_acc"]) or (ba2 == best["both_acc"] and la2 > best["lower_acc"]):
                                best.update({"both_acc": ba2, "lower_acc": la2, "upper_acc": ua2,
                                             "f_upper": fua2, "f_lower": fla2, "f_both": fba2,
                                             "n_raw": n_raw2, "n_filtered": n_f2,
                                             "setup": (mode, tagref, k, b, T_delta, T_ratio, T_abs_pair, True),
                                             "rows": rows2, "pair_abs": pair_abs_fit})

            elif mode == "affine":
                if not (np.isfinite(ref_pos) and np.isfinite(target_pos)):
                    continue
                for tagref, kind, ref_neg in ref_candidates:
                    if not (np.isfinite(ref_neg) and np.isfinite(cutoff_abs_solo)):
                        continue
                    k, b = fit_affine_from_two_points(ref_neg, cutoff_abs_solo, ref_pos, target_pos)
                    if not (np.isfinite(k) and np.isfinite(b)):
                        continue

                    ref_nc_median_adj = float(np.median([apply_calib_values(v, v, "affine", k, b)[0] for v in I_nc_list])) if len(I_nc_list) > 0 else np.nan
                    T_delta = T_delta_base
                    T_ratio = args.t_ratio if np.isfinite(args.t_ratio) else \
                              (float(cutoff_abs_solo / ref_nc_median_adj) if (np.isfinite(ref_nc_median_adj) and ref_nc_median_adj > 0) else np.nan)

                    acc, rows, pair_abs_fit = eval_on_test_all(T_delta, T_ratio, f"affine|{tagref}+soloPos", k, b, np.nan, use_pair_abs=False)
                    if acc:
                        ua, la, ba, fua, fla, fba, n_raw, n_f = acc
                        sweep_rows.append(["affine", f"{tagref}+soloPos", f"{k:.6f}", f"{b:.6f}", T_delta, T_ratio, False, np.nan,
                                           ua, la, ba, fua, fla, fba, n_raw, n_f,
                                           None if pair_abs_fit is None else pair_abs_fit.get("T_abs_pair"),
                                           None if pair_abs_fit is None else pair_abs_fit.get("YoudenJ")])
                        if (ba > best["both_acc"]) or (ba == best["both_acc"] and la > best["lower_acc"]):
                            best.update({"both_acc": ba, "lower_acc": la, "upper_acc": ua,
                                         "f_upper": fua, "f_lower": fla, "f_both": fba,
                                         "n_raw": n_raw, "n_filtered": n_f,
                                         "setup": ("affine", f"{tagref}+soloPos", k, b, T_delta, T_ratio, np.nan, False),
                                         "rows": rows, "pair_abs": pair_abs_fit})

                    if force_use_pair_abs and pair_abs_fit and pair_abs_fit.get("T_abs_pair") is not None:
                        T_abs_pair = float(pair_abs_fit["T_abs_pair"])
                        acc2, rows2, _ = eval_on_test_all(T_delta, T_ratio, f"affine|{tagref}+soloPos", k, b, T_abs_pair, use_pair_abs=True)
                        if acc2:
                            ua2, la2, ba2, fua2, fla2, fba2, n_raw2, n_f2 = acc2
                            sweep_rows.append(["affine", f"{tagref}+soloPos", f"{k:.6f}", f"{b:.6f}", T_delta, T_ratio, True, T_abs_pair,
                                               ua2, la2, ba2, fua2, fla2, fba2, n_raw2, n_f2,
                                               pair_abs_fit.get("T_abs_pair"), pair_abs_fit.get("YoudenJ")])
                            if (ba2 > best["both_acc"]) or (ba2 == best["both_acc"] and la2 > best["lower_acc"]):
                                best.update({"both_acc": ba2, "lower_acc": la2, "upper_acc": ua2,
                                             "f_upper": fua2, "f_lower": fla2, "f_both": fba2,
                                             "n_raw": n_raw2, "n_filtered": n_f2,
                                             "setup": ("affine", f"{tagref}+soloPos", k, b, T_delta, T_ratio, T_abs_pair, True),
                                             "rows": rows2, "pair_abs": pair_abs_fit})

        # 스윕 결과 저장
        sweep_csv = out_dir / f"calibration_sweep_{tag}.csv"
        if sweep_rows:
            with open(sweep_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["mode", "ref", "k", "b", "T_delta", "T_ratio", "use_pair_abs", "T_abs_pair",
                            "UpperACC", "LowerACC", "BothACC", "F_UpperACC", "F_LowerACC", "F_BothACC",
                            "N_raw", "N_filtered", "T_abs_pair_fit", "YoudenJ"])
                for r in sweep_rows:
                    w.writerow(r)

        # 최종 설정 만들기
        if best["setup"] is not None:
            sel_mode, sel_ref, sel_k, sel_b, sel_Td, sel_Tr, sel_Tabs, sel_useAbs = best["setup"]
            selected = {
                "mode": sel_mode, "ref": sel_ref,
                "k": float(sel_k) if np.isfinite(sel_k) else None,
                "b": float(sel_b) if np.isfinite(sel_b) else None,
                "T_delta": float(sel_Td) if np.isfinite(sel_Td) else None,
                "T_ratio": float(sel_Tr) if np.isfinite(sel_Tr) else None,
                "T_abs_pair": float(sel_Tabs) if np.isfinite(sel_Tabs) else None,
                "use_pair_abs": bool(sel_useAbs),
                "UpperACC": float(best["upper_acc"]) if np.isfinite(best["upper_acc"]) else None,
                "LowerACC": float(best["lower_acc"]) if np.isfinite(best["lower_acc"]) else None,
                "BothACC": float(best["both_acc"]) if np.isfinite(best["both_acc"]) else None,
                "F_UpperACC": float(best["f_upper"]) if np.isfinite(best["f_upper"]) else None,
                "F_LowerACC": float(best["f_lower"]) if np.isfinite(best["f_lower"]) else None,
                "F_BothACC": float(best["f_both"]) if np.isfinite(best["f_both"]) else None,
                "N_raw": int(best["n_raw"]), "N_filtered": int(best["n_filtered"])
            }
        else:
            selected = {
                "mode": "none", "ref": "base",
                "k": None, "b": None,
                "T_delta": float(T_delta_base) if np.isfinite(T_delta_base) else None,
                "T_ratio": float(T_ratio_base) if np.isfinite(T_ratio_base) else None,
                "T_abs_pair": None,
                "use_pair_abs": force_use_pair_abs,
                "UpperACC": None, "LowerACC": None, "BothACC": None,
                "F_UpperACC": None, "F_LowerACC": None, "F_BothACC": None,
                "N_raw": 0, "N_filtered": 0
            }

        # pair_rows에 rule 문자열만 갱신 저장(참고용)
        T_delta_final = selected["T_delta"] if selected["T_delta"] is not None else T_delta_base
        T_ratio_final = selected["T_ratio"] if selected["T_ratio"] is not None else T_ratio_base
        T_abs_pair_final = selected["T_abs_pair"] if selected["T_abs_pair"] is not None else np.nan
        use_pair_abs_final = bool(selected.get("use_pair_abs", False))

        for r in pair_rows:
            try:
                Iu = float(r["I_nc"]); Il = float(r["I_sample"])
            except Exception:
                continue
            rule_parts = []
            if np.isfinite(T_delta_final):
                rule_parts.append(f"Δ%≥{T_delta_final:.1f}")
            if np.isfinite(T_ratio_final):
                rule_parts.append(f"ratio≥{T_ratio_final:.3f}×")
            if use_pair_abs_final and np.isfinite(T_abs_pair_final):
                rule_parts.append(f"Il≥{T_abs_pair_final:.1f}")
            r["rule"] = " OR ".join(rule_parts)
            r["pred_upper"] = ""  # pair 자체는 GT 없음; 예측은 test_all에서

        with open(out_dir / f"pair_analysis_{tag}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["image_path", "I_nc", "I_sample", "delta_pct", "ratio", "rule", "pred_upper", "pred_lower", "note"])
            w.writeheader()
            for r in pair_rows:
                w.writerow(r)

        # test_all 최종 표 저장 (선택 규칙으로 재평가) — Iu/Il 포함
        _, rows_final, _ = eval_on_test_all(T_delta_final, T_ratio_final, selected["mode"] + "|final",
                                            selected["k"] if selected["k"] is not None else np.nan,
                                            selected["b"] if selected["b"] is not None else np.nan,
                                            T_abs_pair_final, use_pair_abs_final)
        test_csv = out_dir / f"test_all_report_{tag}.csv"
        with open(test_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "image_path", "upper_exp", "lower_exp", "upper_pred", "lower_pred",
                "Iu", "Il", "delta_pct", "ratio", "note",
                "calib", "k", "b", "T_delta", "T_ratio", "T_abs_pair", "use_pair_abs"
            ])
            for r in rows_final:
                w.writerow(r)

        # intensity 그룹 CSV(고정 파일은 그대로 유지; 시나리오 공통)
        dist_csv = out_dir / "intensity_groups.csv"
        if not Path(dist_csv).exists():
            with open(dist_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["group", "intensity_a.u."])
                for v in solo_neg_vals:
                    if np.isfinite(v): w.writerow(["solo_neg(G_p95)", v])
                for v in solo_pos_vals:
                    if np.isfinite(v): w.writerow(["solo_pos(G_p95)", v])
                for v in I_nc_list:
                    if np.isfinite(v): w.writerow(["pair_neg_upper(G_p95)", v])
                for v in I_sm_list:
                    if np.isfinite(v): w.writerow(["pair_pos_lower(G_p95)", v])

        # 요약
        T_abs_pair_fit = best["pair_abs"]["T_abs_pair"] if (best["pair_abs"] and best["pair_abs"].get("T_abs_pair") is not None) else None
        summary = {
            "solo_neg_median": float(median_solo_neg) if np.isfinite(median_solo_neg) else None,
            "solo_pos_median": float(median_solo_pos) if np.isfinite(median_solo_pos) else None,
            "pair_nc_median":  float(median_pair_nc)  if np.isfinite(median_pair_nc)  else None,
            "pair_sm_median":  float(median_pair_sm)  if np.isfinite(median_pair_sm)  else None,
            "cutoff_abs_solo": float(cutoff_abs_solo),
            "cutoff_abs_pair_opt": float(T_abs_pair_fit) if T_abs_pair_fit is not None else None,
        }
        with open(out_dir / f"pair_solo_summary_{tag}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            for k, v in summary.items():
                w.writerow([k, v])

        # best_calibration 저장
        best_json = {
            "selected": selected,
            "calibration_modes_tried": ["none", "ratio", "shift", "affine"] if args.calib_auto == "on" else [args.calib],
            "ref_candidates": {
                "mean_solo_neg": mean_solo_neg if np.isfinite(mean_solo_neg) else None,
                "median_solo_neg": median_solo_neg if np.isfinite(median_solo_neg) else None,
                "median_pair_nc": median_pair_nc if np.isfinite(median_pair_nc) else None,
                "blend_median": float(np.median([median_solo_neg, median_pair_nc])) if (np.isfinite(median_solo_neg) and np.isfinite(median_pair_nc)) else None,
                "median_solo_pos": median_solo_pos if np.isfinite(median_solo_pos) else None,
                "R_sp": (median_solo_pos/median_solo_neg) if (np.isfinite(median_solo_pos) and np.isfinite(median_solo_neg) and median_solo_neg>0) else None,
                "target_pos": target_pos if np.isfinite(target_pos) else None
            },
            "final_thresholds": {
                "T_delta": selected["T_delta"],
                "T_ratio": selected["T_ratio"],
                "T_abs_pair": selected["T_abs_pair"],
                "use_pair_abs": selected["use_pair_abs"]
            },
            "files": {
                "solo_analysis_csv": str(solo_csv),
                "pair_analysis_csv": str(out_dir / f"pair_analysis_{tag}.csv"),
                "test_all_report_csv": str(out_dir / f"test_all_report_{tag}.csv"),
                "intensity_groups_csv": str(out_dir / "intensity_groups.csv"),
                "calibration_sweep_csv": str(out_dir / f"calibration_sweep_{tag}.csv"),
                "pair_solo_summary_csv": str(out_dir / f"pair_solo_summary_{tag}.csv")
            },
            "config_fixed": {
                "conf": args.conf, "iou": args.iou, "imgsz": args.imgsz, "method": "G", "metric": "p95"
            }
        }
        with open(out_dir / f"best_calibration_{tag}.json", "w", encoding="utf-8") as f:
            json.dump(best_json, f, ensure_ascii=False, indent=2)

        # metrics_summary (간단 버전)
        metrics = {
            "config": {
                "weights": str(args.weights),
                "conf_fixed": args.conf, "iou": args.iou, "imgsz": args.imgsz,
                "method": "G", "metric": "p95",
                "cutoff_neg_abs_solo": cutoff_abs_solo,
                "use_pair_abs": selected["use_pair_abs"],
                "T_abs_pair": selected["T_abs_pair"],
                "T_ratio": selected["T_ratio"],
                "T_delta": selected["T_delta"]
            },
            "pair_stats": {"median_pair_nc": median_pair_nc},
            "accuracy": {
                "raw": {"upper_acc": selected["UpperACC"], "lower_acc": selected["LowerACC"], "both_acc": selected["BothACC"], "n": selected["N_raw"]},
                "filtered": {"upper_acc": selected["F_UpperACC"], "lower_acc": selected["F_LowerACC"], "both_acc": selected["F_BothACC"], "n": selected["N_filtered"]}
            },
            "files": {
                "pair_analysis_csv": str(out_dir / f"pair_analysis_{tag}.csv"),
                "test_all_report_csv": str(out_dir / f"test_all_report_{tag}.csv")
            }
        }
        with open(out_dir / f"metrics_summary_{tag}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        return selected

    # ---- 실행 모드: 단일 / 프리셋 비교 ----
    comparison_rows = []

    def add_compare_row(tag, sel):
        comparison_rows.append([
            tag,
            sel.get("T_ratio"), sel.get("T_delta"), sel.get("T_abs_pair"), sel.get("use_pair_abs"),
            sel.get("UpperACC"), sel.get("LowerACC"), sel.get("BothACC"),
            sel.get("F_UpperACC"), sel.get("F_LowerACC"), sel.get("F_BothACC"),
            sel.get("N_raw"), sel.get("N_filtered")
        ])

    if args.compare_presets == "on":
        # A: ratio only
        selA = run_one_setup("A", float("nan"), False); add_compare_row("A", selA)

        # B: ratio + Δ% (list)
        tlist = [s.strip() for s in str(args.t_delta_list).split(",") if s.strip() != ""]
        for td in tlist:
            try:
                tval = float(td)
            except Exception:
                continue
            # 간단 태그: B05, B075, B10...
            base = td.replace(".", "")
            tag = f"B{base}" if base else f"B{int(round(tval))}"
            selB = run_one_setup(tag.upper(), tval, False); add_compare_row(tag.upper(), selB)

        # C: ratio + Il(Youden)
        selC = run_one_setup("C", float("nan"), True); add_compare_row("C", selC)

        # D: ratio + Δ% + Il(Youden)
        for td in tlist:
            try:
                tval = float(td)
            except Exception:
                continue
            base = td.replace(".", "")
            tag = f"D{base}" if base else f"D{int(round(tval))}"
            selD = run_one_setup(tag.upper(), tval, True); add_compare_row(tag.upper(), selD)

        with open(out_dir / "scenarios_comparison.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Scenario", "T_ratio", "T_delta", "T_abs_pair", "use_pair_abs",
                        "UpperACC", "LowerACC", "BothACC", "F_UpperACC", "F_LowerACC", "F_BothACC", "N_raw", "N_filtered"])
            for r in comparison_rows:
                w.writerow(r)

    else:
        # 단일 실행 — tag=BASE
        sel = run_one_setup("BASE", args.t_delta, args.use_pair_abs == "on")
        add_compare_row("BASE", sel)
        with open(out_dir / "scenarios_comparison.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Scenario", "T_ratio", "T_delta", "T_abs_pair", "use_pair_abs",
                        "UpperACC", "LowerACC", "BothACC", "F_UpperACC", "F_LowerACC", "F_BothACC", "N_raw", "N_filtered"])
            for r in comparison_rows:
                w.writerow(r)

if __name__ == "__main__":
    main()
