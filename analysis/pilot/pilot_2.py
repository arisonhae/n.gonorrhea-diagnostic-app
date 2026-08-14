# test_2.py
# Blue 촬영 사진(positive/negative)을 배율별로 분석하여
# - 자동 ROI 추출 + 오버레이/신호맵 저장
# - 배율별 집계(ROI 성공률, 분리도-Cohen d, SNR, 포화율, t-test p값)
# - 1배율(1.0x)은 "분석 제외"로 명시
# - summary.json 저장 + previews/<mag>/<pos|neg>/*_overlay.jpg, *_signal.jpg 저장

import cv2
import json
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageOps
import re
import math
from scipy.stats import ttest_ind

# ================== 환경에 맞게 경로 수정 ==================
BASE_DIR = r"C:\n.gonorrhea_diagnostic_app\pilotimage_2"
# =========================================================

CLEAN_PREVIEWS = True
EXCLUDE_MAGS_MANUAL = {1.0}        # 1배율은 수동 제외
ROI_SUCCESS_THRESHOLD = 0.0        # 자동 제외 비활성화 (원하면 0.5~0.7)

ANALYSIS_ROOT = Path(BASE_DIR) / "_analysis"
PREVIEW_ROOT  = ANALYSIS_ROOT / "previews"
SUMMARY_JSON  = ANALYSIS_ROOT / "summary.json"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAG_PAT = re.compile(r"_(\d+)(?:p(\d+))?x_", re.IGNORECASE)

def parse_mag_from_name(name: str):
    m = MAG_PAT.search(name)
    if not m: return None
    a = int(m.group(1)); b = m.group(2)
    return float(a) if b is None else float(f"{a}.{b}")

def imread_safe(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is not None: return bgr
    try:
        pil = Image.open(path)
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        arr = np.array(pil)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None

def resize_keep(bgr, max_side=1200):
    h, w = bgr.shape[:2]
    if max(h, w) <= max_side: return bgr
    s = max_side / max(h, w)
    return cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def auto_signal_map(bgr):
    f = bgr.astype(np.float32)
    B, G, R = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    eps = 1e-6
    alpha = np.median(G) / (np.median(R) + eps)
    cand1 = G - alpha * R - 0.15 * B
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    bg = cv2.medianBlur(Y.astype(np.uint8), 31).astype(np.float32)
    cand2 = Y - bg
    cand3 = G

    def normalize_u8(S):
        p1, p99 = np.percentile(S, 1), np.percentile(S, 99)
        Sn = (S - p1) / max(p99 - p1, 1e-6)
        return (np.clip(Sn, 0, 1) * 255).astype(np.uint8)

    def safe_score(Sn8):
        Sn8 = Sn8.astype(np.float32)
        top = float(np.percentile(Sn8, 99.5))
        med = float(np.median(Sn8))
        low = Sn8 < np.percentile(Sn8, 50)
        low_std = float(np.std(Sn8[low])) if np.any(low) else float(np.std(Sn8))
        return (top - med) / (low_std + 1e-6)

    best_map, best_score = None, -1e9
    for S in (cand1, cand2, cand3):
        Sn8 = normalize_u8(S)
        sc = safe_score(Sn8)
        if sc > best_score:
            best_score, best_map = sc, Sn8
    return best_map if best_map is not None else normalize_u8(G)

def extract_roi_masks(sig_u8, max_regions=2):
    if sig_u8 is None or sig_u8.size == 0: return []
    blur = cv2.GaussianBlur(sig_u8, (5, 5), 1.2)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=1)
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return []
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:max_regions]
    masks = []
    for c in cnts:
        m = np.zeros_like(sig_u8, np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        masks.append(m)
    return masks

def analyze_one_image(img_path: Path, save_dir: Path):
    bgr0 = imread_safe(img_path)
    if bgr0 is None:
        raise RuntimeError("이미지 로딩 실패")
    bgr = resize_keep(bgr0, 1200)
    sig = auto_signal_map(bgr)
    masks = extract_roi_masks(sig, max_regions=2)
    if not masks:
        raise RuntimeError("ROI 검출 실패")

    roi_mask = np.zeros_like(sig)
    for m in masks: roi_mask = cv2.bitwise_or(roi_mask, m)
    bg_mask = cv2.bitwise_not(roi_mask)

    roi_vals = sig[roi_mask > 0].astype(np.float32)
    bg_vals  = sig[bg_mask  > 0].astype(np.float32)
    if roi_vals.size < 20 or bg_vals.size < 50:
        raise RuntimeError("ROI/배경 픽셀 부족")

    sig_mean = float(np.mean(roi_vals))
    bg_mean  = float(np.mean(bg_vals))
    bg_std   = float(np.std(bg_vals) + 1e-6)
    snr      = (sig_mean - bg_mean) / bg_std
    sat      = float(np.mean(roi_vals >= 250.0))

    # 프리뷰 저장
    save_dir.mkdir(parents=True, exist_ok=True)
    overlay = bgr.copy()
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(str(save_dir / f"{img_path.stem}_overlay.jpg"), overlay)
    cv2.imwrite(str(save_dir / f"{img_path.stem}_signal.jpg"), sig)

    return {
        "filename": img_path.name,
        "sig_mean": sig_mean,
        "bg_mean": bg_mean,
        "bg_std":  bg_std,
        "snr":     float(snr),
        "sat":     float(sat)
    }

def list_images(dir_path: Path):
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() in IMG_EXTS])

def cohen_d(pos_vals, neg_vals):
    if len(pos_vals) == 0 or len(neg_vals) == 0:
        return float("nan")
    pos_vals = np.array(pos_vals, dtype=np.float32)
    neg_vals = np.array(neg_vals, dtype=np.float32)
    pos_std = np.std(pos_vals) if len(pos_vals) > 1 else 1.0
    neg_std = np.std(neg_vals) if len(neg_vals) > 1 else 1.0
    pooled_std = math.sqrt((pos_std**2 + neg_std**2) / 2.0)
    if pooled_std == 0: return float("nan")
    return float((np.mean(pos_vals) - np.mean(neg_vals)) / pooled_std)

def main():
    pos_dir = Path(BASE_DIR) / "positive"
    neg_dir = Path(BASE_DIR) / "negative"

    print("=== 경로 점검 ===")
    print(f"{pos_dir}: {'OK' if pos_dir.is_dir() else 'MISSING'}")
    print(f"{neg_dir}: {'OK' if neg_dir.is_dir() else 'MISSING'}\n")

    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    if CLEAN_PREVIEWS and PREVIEW_ROOT.exists():
        shutil.rmtree(PREVIEW_ROOT)
    PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)

    pos_files = list_images(pos_dir)
    neg_files = list_images(neg_dir)

    pos_by_mag, neg_by_mag = {}, {}
    for p in pos_files:
        m = parse_mag_from_name(p.name)
        if m is not None: pos_by_mag.setdefault(m, []).append(p)
    for p in neg_files:
        m = parse_mag_from_name(p.name)
        if m is not None: neg_by_mag.setdefault(m, []).append(p)

    all_mags = sorted(set(pos_by_mag.keys()) | set(neg_by_mag.keys()))
    print("=== 배율/파일 개수 ===")
    for m in all_mags:
        print(f"mag {m}: pos={len(pos_by_mag.get(m, []))}, neg={len(neg_by_mag.get(m, []))}")
    print("\n=== 분석 시작 ===")

    table_by_mag = {}
    fails = []

    for m in all_mags:
        pos_list, neg_list = [], []
        pos_prev = PREVIEW_ROOT / f"{m:.1f}" / "positive"
        neg_prev = PREVIEW_ROOT / f"{m:.1f}" / "negative"

        for p in pos_by_mag.get(m, []):
            try: pos_list.append(analyze_one_image(p, pos_prev))
            except Exception as e: fails.append((p.name, str(e)))

        for p in neg_by_mag.get(m, []):
            try: neg_list.append(analyze_one_image(p, neg_prev))
            except Exception as e: fails.append((p.name, str(e)))

        n_pos_total = len(pos_by_mag.get(m, []))
        n_neg_total = len(neg_by_mag.get(m, []))
        n_pos_valid = len(pos_list)
        n_neg_valid = len(neg_list)
        total = n_pos_valid + n_neg_valid

        roi_success = 0.0
        denom = n_pos_total + n_neg_total
        if denom > 0:
            roi_success = total / denom

        pos_sig = [r["sig_mean"] for r in pos_list]
        neg_sig = [r["sig_mean"] for r in neg_list]

        sep = cohen_d(pos_sig, neg_sig)
        try:
            pval = float(ttest_ind(pos_sig, neg_sig, equal_var=False).pvalue) \
                   if len(pos_sig) >= 2 and len(neg_sig) >= 2 else None
        except Exception:
            pval = None

        pos_snr_med = float(np.median([r["snr"] for r in pos_list])) if pos_list else None
        saturation  = float(np.mean([r["sat"] for r in (pos_list + neg_list)])) if (pos_list or neg_list) else None

        table_by_mag[f"{m:.1f}"] = {
            "mag": float(m),
            "n_pos": n_pos_total, "n_neg": n_neg_total,
            "n_pos_valid": n_pos_valid, "n_neg_valid": n_neg_valid,
            "roi_success": float(roi_success),
            "sep_cohen_d": None if np.isnan(sep) else float(sep),
            "sep_p_value": pval,
            "pos_snr_med": pos_snr_med,
            "saturation": saturation,
            "excluded": False, "reason": ""
        }

    # 제외 정책
    excluded_manual, excluded_auto = [], []
    for k, row in table_by_mag.items():
        m = float(k)
        if m in EXCLUDE_MAGS_MANUAL:
            row["excluded"] = True; row["reason"] = "manual_1x_poor_roi"; excluded_manual.append(m)
        elif ROI_SUCCESS_THRESHOLD > 0.0 and (row.get("roi_success", 0.0) < ROI_SUCCESS_THRESHOLD):
            row["excluded"] = True; row["reason"] = "low_roi_success"; excluded_auto.append(m)

    used_rows = {k: v for k, v in table_by_mag.items() if not v.get("excluded", False)}

    # 승자 선정
    best_sep_mag, best_sep_val = None, -1e9
    for k, v in used_rows.items():
        d = v.get("sep_cohen_d")
        if d is None: continue
        if d > best_sep_val: best_sep_val, best_sep_mag = d, float(k)

    best_snr_mag, best_snr_val = None, -1e9
    for k, v in used_rows.items():
        s = v.get("pos_snr_med")
        if s is None: continue
        # 포화율 페널티(높을수록 감점)
        sat = v.get("saturation")
        score = s if sat is None else s * (1.0 - min(max(sat, 0.0), 1.0) * 0.3)
        if score > best_snr_val: best_snr_val, best_snr_mag = score, float(k)

    summary = {
        "base_dir": str(BASE_DIR),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exclusion": {
            "manual": sorted(excluded_manual),
            "auto":   sorted(excluded_auto),
            "used_magnifications": sorted([float(k) for k in used_rows.keys()])
        },
        "table_by_mag": table_by_mag,
        "winners": {
            "best_separation_mag": best_sep_mag,
            "best_separation_val": None if best_sep_mag is None else float(best_sep_val),
            "best_pos_snr_mag": best_snr_mag,
            "best_pos_snr_val": None if best_snr_mag is None else float(best_snr_val)
        },
        "fails": fails,
        "preview_root": str(PREVIEW_ROOT)
    }

    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== 완료 ===")
    print(f"[saved] {SUMMARY_JSON}")
    print(f"[preview dir] {PREVIEW_ROOT}")

if __name__ == "__main__":
    main()

# python pilot_2.py