# fluorescence_test_1.py
# 폴더에서 이미지를 읽어 Blue/White 파장 성능 비교
# - 자동 신호맵(파장 무관) + 자동 ROI 추출
# - ROI 오버레이/신호맵 저장
# - 요약 지표/추천 파장 summary.json 저장

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageOps

# ================== 환경에 맞게 경로 수정 ==================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import PILOT_1_WAVELENGTH, OUT_PILOT
BASE_DIR = str(PILOT_1_WAVELENGTH)
# ==========================================================

# --------- 유틸 ---------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def imread_safe(path: Path):
    """cv2.imread 실패 시 PIL(EXIF 회전 보정)로 안전 로딩"""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is not None:
        return bgr
    try:
        pil = Image.open(path)
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        arr = np.array(pil)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        return None

def resize_keep(bgr, max_side=1200):
    h, w = bgr.shape[:2]
    if max(h, w) <= max_side:
        return bgr
    s = max_side / max(h, w)
    return cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

# --------- 신호맵 & ROI (NaN/빈배열 방어 포함) ---------
def auto_signal_map(bgr):
    """파장 무관 자동 신호맵 (Blue/White 모두 커버, NaN 방어)"""
    f = bgr.astype(np.float32)
    B, G, R = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    eps = 1e-6

    # 후보1: Blue 배경 가정(G - a*R - 0.15*B)
    alpha = np.median(G) / (np.median(R) + eps)
    cand1 = G - alpha * R - 0.15 * B

    # 후보2: White 배경 가정(밝기 Y에서 롤링 배경 제거)
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    bg = cv2.medianBlur(Y.astype(np.uint8), 31).astype(np.float32)
    cand2 = Y - bg

    # 후보3: G 채널
    cand3 = G

    def normalize_u8(S):
        p1, p99 = np.percentile(S, 1), np.percentile(S, 99)
        Sn = (S - p1) / max(p99 - p1, 1e-6)
        Sn = np.clip(Sn, 0, 1)
        return (Sn * 255).astype(np.uint8)

    def safe_score(Sn):
        Sn = Sn.astype(np.float32)
        top = float(np.percentile(Sn, 99.5))
        med = float(np.median(Sn))
        low_mask = Sn < np.percentile(Sn, 50)
        if np.any(low_mask):
            low_std = float(np.std(Sn[low_mask]))
        else:
            low_std = float(np.std(Sn))
        return (top - med) / (low_std + 1e-6)

    best_map, best_score = None, -1e9
    for S in (cand1, cand2, cand3):
        Sn8 = normalize_u8(S)
        sc = safe_score(Sn8)
        if sc > best_score:
            best_score, best_map = sc, Sn8

    if best_map is None:
        best_map = normalize_u8(G)  # 최후의 보루
    return best_map

def extract_roi_masks(sig_u8, max_regions=2):
    """신호맵에서 상위 블롭 ROI 마스크 추출 (빈 입력 방어)"""
    if sig_u8 is None or sig_u8.size == 0:
        return []

    blur = cv2.GaussianBlur(sig_u8, (5, 5), 1.2)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=1)
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:max_regions]
    masks = []
    for c in cnts:
        m = np.zeros_like(sig_u8, np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        masks.append(m)
    return masks

# --------- 단일 이미지 분석 ---------
def analyze_one_image(img_path: Path, save_preview_dir: Path):
    print(f"[DEBUG] Processing: {img_path.name}")
    
    bgr0 = imread_safe(img_path)
    if bgr0 is None:
        raise RuntimeError("이미지 로딩 실패")
    
    print(f"[DEBUG] Image loaded: shape={bgr0.shape}, dtype={bgr0.dtype}")
    
    bgr = resize_keep(bgr0, 1200)
    print(f"[DEBUG] After resize: shape={bgr.shape}")

    # 신호맵 & ROI
    sig = auto_signal_map(bgr)
    print(f"[DEBUG] Signal map: shape={sig.shape if sig is not None else 'None'}, "
          f"dtype={sig.dtype if sig is not None else 'None'}, "
          f"size={sig.size if sig is not None else 0}")
    
    if sig is None or sig.size == 0:
        raise RuntimeError("신호맵 생성 실패 - 빈 배열")
    
    masks = extract_roi_masks(sig, max_regions=2)
    if not masks:
        raise RuntimeError("ROI 검출 실패")

    roi_mask = np.zeros_like(sig)
    for m in masks:
        roi_mask = cv2.bitwise_or(roi_mask, m)
    bg_mask = cv2.bitwise_not(roi_mask)

    roi_vals = sig[roi_mask > 0].astype(np.float32)
    bg_vals = sig[bg_mask > 0].astype(np.float32)
    if roi_vals.size < 20 or bg_vals.size < 50:
        raise RuntimeError("ROI/배경 픽셀 부족")

    sig_mean = float(np.mean(roi_vals))
    bg_mean = float(np.mean(bg_vals))
    bg_std = float(np.std(bg_vals) + 1e-6)
    snr = (sig_mean - bg_mean) / bg_std
    sat = float(np.mean(roi_vals >= 250.0))

    # 오버레이 & 신호맵 저장
    overlay = bgr.copy()
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    save_preview_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_preview_dir / f"{img_path.stem}_overlay.jpg"), overlay)
    cv2.imwrite(str(save_preview_dir / f"{img_path.stem}_signal.jpg"), sig)

    return {
        "filename": img_path.name,
        "sig_mean": sig_mean,
        "bg_mean": bg_mean,
        "bg_std": bg_std,
        "snr": float(snr),
        "sat": float(sat),
    }

# --------- 배치 실행 ---------
def list_images(dir_path: Path):
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() in IMG_EXTS])

def run_batch(files, preview_root: Path, tag: str):
    results, fails = [], []
    prev_dir = preview_root / tag
    prev_dir.mkdir(parents=True, exist_ok=True)

    for p in files:
        try:
            r = analyze_one_image(p, prev_dir)
            results.append(r)
        except Exception as e:
            fails.append((p.name, str(e)))
            # 실패한 것도 어디까지 읽혔는지 보려면 여기서 원본 축소본 저장해도 됨
    return results, fails

# --------- 집계/점수 ---------
def aggregate_metrics(pos_results, neg_results):
    if not pos_results:
        return None
    pos_mean = float(np.median([r["sig_mean"] for r in pos_results]))
    neg_mean = float(np.median([r["sig_mean"] for r in neg_results])) if neg_results else 0.0
    all_res = pos_results + neg_results
    pooled = float(np.median([r["bg_std"] for r in all_res])) if all_res else 1.0
    separation = (pos_mean - neg_mean) / (pooled + 1e-6)
    snr_med = float(np.median([r["snr"] for r in pos_results]))
    sat_mean = float(np.mean([r["sat"] for r in all_res])) if all_res else 0.0
    return {
        "valid_pos": len(pos_results),
        "valid_neg": len(neg_results),
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "separation": float(separation),
        "snr": snr_med,
        "saturation": float(sat_mean),  # 0~1
    }

def score_rule(m):
    # 포화 페널티 강하게, 분리도/신호품질 가중
    return 0.6 * m["snr"] + 1.2 * m["separation"] - 8.0 * max(m["saturation"] - 0.02, 0.0)

# --------- 메인 ---------
def main():
    base = Path(BASE_DIR)
    blue_pos  = base / "wavelength_blue"  / "positive"
    blue_neg  = base / "wavelength_blue"  / "negative"
    white_pos = base / "wavelength_white" / "positive"
    white_neg = base / "wavelength_white" / "negative"

    # 경로/파일 점검
    print("=== 경로 점검 ===")
    for d in (blue_pos, blue_neg, white_pos, white_neg):
        print(f"{d}: {'OK' if d.is_dir() else 'MISSING'}")
    print()

    bp_files = list_images(blue_pos)
    bn_files = list_images(blue_neg)
    wp_files = list_images(white_pos)
    wn_files = list_images(white_neg)

    print("=== 파일 개수 ===")
    print(f"Blue Positive : {len(bp_files)}")
    print(f"Blue Negative : {len(bn_files)}")
    print(f"White Positive: {len(wp_files)}")
    print(f"White Negative: {len(wn_files)}")
    print()

    analysis_root = OUT_PILOT / "pilot_1_wavelength"
    preview_root = analysis_root / "previews"
    analysis_root.mkdir(exist_ok=True, parents=True)
    preview_root.mkdir(exist_ok=True, parents=True)

    # 실행
    print("=== Blue/White 분석 시작 ===")
    bp_res, bp_fail = run_batch(bp_files, preview_root, "blue_pos")
    bn_res, bn_fail = run_batch(bn_files, preview_root, "blue_neg")
    wp_res, wp_fail = run_batch(wp_files, preview_root, "white_pos")
    wn_res, wn_fail = run_batch(wn_files, preview_root, "white_neg")

    # 실패 로그
    for name, why in (bp_fail + bn_fail + wp_fail + wn_fail):
        print(f"[FAIL] {name} -> {why}")

    # 집계
    blue_metrics  = aggregate_metrics(bp_res, bn_res)
    white_metrics = aggregate_metrics(wp_res, wn_res)

    result = {
        "base_dir": str(base),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "counts": {
            "blue_pos": len(bp_files),
            "blue_neg": len(bn_files),
            "white_pos": len(wp_files),
            "white_neg": len(wn_files),
        },
        "metrics": {"blue": blue_metrics, "white": white_metrics},
        "scores": {},
        "winner": None,
        "fails": {
            "blue_pos": bp_fail,
            "blue_neg": bn_fail,
            "white_pos": wp_fail,
            "white_neg": wn_fail,
        },
        "preview_dir": str(preview_root),
    }

    if blue_metrics:
        result["scores"]["blue"] = score_rule(blue_metrics)
    if white_metrics:
        result["scores"]["white"] = score_rule(white_metrics)

    if blue_metrics and white_metrics:
        result["winner"] = "Blue" if result["scores"]["blue"] > result["scores"]["white"] else "White"
    elif blue_metrics:
        result["winner"] = "Blue"
    elif white_metrics:
        result["winner"] = "White"

    # 저장
    out_json = analysis_root / "summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n=== 완료 ===")
    print(f"[saved] {out_json}")
    print(f"[preview dir] {preview_root}")

if __name__ == "__main__":
    main()