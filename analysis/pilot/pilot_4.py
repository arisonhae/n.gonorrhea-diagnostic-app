# pilot_4.py
# 조명 밝기 강도별 형광값 비교 분석
# - intensity_low, intensity_medium, intensity_high 폴더 분석
# - 각 밝기별로 positive/negative 분리도(Cohen's d) + Welch t-test(p-value)
# - 이전 파일럿과 동일한 ROI/신호맵 로직 유지
# - 프리뷰(overlay/signal) 저장

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageOps

# scipy가 없어도 전체 파이프라인이 동작하도록 안전 import
try:
    from scipy import stats
except Exception:
    stats = None

# ===== 설정 =====
BASE_DIR = r"C:\n.gonorrhea_diagnostic_app\pilotimage_4"

# ===== 공통 유틸 =====
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

# ===== 신호맵 & ROI (이전과 동일) =====
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
        low_std = float(np.std(Sn[low_mask])) if np.any(low_mask) else float(np.std(Sn))
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
    """신호맵에서 상위 블롭 ROI 마스크 추출"""
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

# ===== 단일 이미지 분석 =====
def analyze_one_image(img_path: Path, save_preview_dir: Path):
    """ROI 내 형광 통계 + 프리뷰 저장"""
    bgr0 = imread_safe(img_path)
    if bgr0 is None:
        raise RuntimeError("이미지 로딩 실패")
    bgr = resize_keep(bgr0, 1200)

    sig = auto_signal_map(bgr)
    if sig is None or sig.size == 0:
        raise RuntimeError("신호맵 생성 실패")

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
    sig_median = float(np.median(roi_vals))
    sig_max = float(np.max(roi_vals))
    bg_mean = float(np.mean(bg_vals))
    bg_std = float(np.std(bg_vals) + 1e-6)
    snr = float((sig_mean - bg_mean) / bg_std)
    sat = float(np.mean(roi_vals >= 250.0))

    # 프리뷰 저장
    save_preview_dir.mkdir(parents=True, exist_ok=True)
    overlay = bgr.copy()
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(str(save_preview_dir / f"{img_path.stem}_overlay.jpg"), overlay)
    cv2.imwrite(str(save_preview_dir / f"{img_path.stem}_signal.jpg"), sig)

    return {
        "filename": img_path.name,
        "sig_mean": sig_mean,
        "sig_median": sig_median,
        "sig_max": sig_max,
        "bg_mean": bg_mean,
        "bg_std": bg_std,
        "snr": snr,
        "sat": sat,
    }

# ===== 튜브별 처리 =====
def list_images(dir_path: Path):
    if not dir_path.exists():
        return []
    return sorted([p for p in dir_path.iterdir() if p.suffix.lower() in IMG_EXTS])

def extract_tube_id(filename: str):
    """tube_1_1.jpg → tube_1  (튜브 단위 평균을 위해)"""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return stem

def process_intensity_category(category_dir: Path, preview_dir: Path):
    """
    한 intensity/category 폴더 처리.
    - 튜브별로 이미지(보통 2장)를 평균 내어 대표값 생성
    """
    tube_results = {}
    fails = []

    for img_path in list_images(category_dir):
        tube_id = extract_tube_id(img_path.name)
        try:
            r = analyze_one_image(img_path, preview_dir)
            tube_results.setdefault(tube_id, []).append(r)
        except Exception as e:
            fails.append((img_path.name, str(e)))

    tube_averages = {}
    for tube_id, rs in tube_results.items():
        tube_averages[tube_id] = {
            "sig_mean": float(np.mean([x["sig_mean"] for x in rs])),
            "sig_median": float(np.median([x["sig_mean"] for x in rs])),
            "snr": float(np.mean([x["snr"] for x in rs])),
            "image_count": len(rs),
        }

    return tube_averages, fails

# ===== 분리도/유의성 =====
def calculate_separation(pos_tubes: dict, neg_tubes: dict):
    """Cohen's d + Welch t-test (샘플 수/분산 다름 가정)"""
    if not pos_tubes or not neg_tubes:
        return None

    pos_vals = np.array([v["sig_mean"] for v in pos_tubes.values()], dtype=float)
    neg_vals = np.array([v["sig_mean"] for v in neg_tubes.values()], dtype=float)

    pos_mean = float(np.mean(pos_vals))
    neg_mean = float(np.mean(neg_vals))
    pos_std = float(np.std(pos_vals, ddof=1)) if pos_vals.size > 1 else 0.0
    neg_std = float(np.std(neg_vals, ddof=1)) if neg_vals.size > 1 else 0.0

    # Cohen's d (pooled via 두 집단 표준편차 평균)
    pooled_std = float(np.sqrt((pos_std**2 + neg_std**2) / 2.0))
    d = (pos_mean - neg_mean) / (pooled_std + 1e-6)

    # Welch t-test
    if stats and (pos_vals.size >= 2 and neg_vals.size >= 2):
        t_stat, p_value = stats.ttest_ind(pos_vals, neg_vals, equal_var=False)
        p_value = float(p_value)
    else:
        p_value = None

    return {
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "pos_std": pos_std,
        "neg_std": neg_std,
        "cohens_d": float(d),
        "p_value": p_value,
        "pos_count": int(pos_vals.size),
        "neg_count": int(neg_vals.size),
    }

# ===== 메인 =====
def main():
    base = Path(BASE_DIR)
    print("=" * 60)
    print("조명 밝기 강도별 형광 분석")
    print("=" * 60)

    # 순서를 고정(low → medium → high)
    wanted = ["intensity_low", "intensity_medium", "intensity_high"]
    intensity_folders = [base / w for w in wanted if (base / w).is_dir()]
    if not intensity_folders:
        # fallback: intensity_* 전부
        intensity_folders = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("intensity_")])

    print(f"\n발견된 밝기 단계: {[d.name for d in intensity_folders]}")

    # 분석 폴더
    analysis_root = base / "_analysis"
    preview_root = analysis_root / "previews"
    analysis_root.mkdir(exist_ok=True, parents=True)
    preview_root.mkdir(exist_ok=True, parents=True)

    all_results = {}
    all_fails = []

    for intensity_dir in intensity_folders:
        intensity_name = intensity_dir.name.replace("intensity_", "")
        print(f"\n[처리 중] {intensity_name.upper()}")

        pos_dir = intensity_dir / "positive"
        neg_dir = intensity_dir / "negative"

        prev_pos = (preview_root / intensity_name / "positive")
        prev_neg = (preview_root / intensity_name / "negative")

        pos_tubes, pos_fails = process_intensity_category(pos_dir, prev_pos)
        neg_tubes, neg_fails = process_intensity_category(neg_dir, prev_neg)

        print(f"  - Positive 튜브: {len(pos_tubes)}")
        print(f"  - Negative 튜브: {len(neg_tubes)}")

        all_fails.extend([(f[0], intensity_name, "positive", f[1]) for f in pos_fails])
        all_fails.extend([(f[0], intensity_name, "negative", f[1]) for f in neg_fails])

        separation = calculate_separation(pos_tubes, neg_tubes)
        if separation:
            pv = separation["p_value"]
            pv_str = f"{pv:.4f}" if pv is not None else "N/A"
            print(f"  - Cohen's d: {separation['cohens_d']:.3f}, p-value: {pv_str}")

        all_results[intensity_name] = {
            "positive": pos_tubes,
            "negative": neg_tubes,
            "separation": separation,
        }

    # 실패 로그
    if all_fails:
        print("\n[실패한 이미지]")
        for fname, intensity, category, reason in all_fails:
            print(f"  {intensity}/{category}/{fname}: {reason}")

    # 최적 밝기 (Cohen's d 최대)
    best_intensity, best_separation = None, -1e9
    for intensity_name, dct in all_results.items():
        sep = dct.get("separation")
        if sep and sep["cohens_d"] > best_separation:
            best_separation = sep["cohens_d"]
            best_intensity = intensity_name

    print("\n" + "=" * 60)
    print("최종 결과")
    print("=" * 60)
    if best_intensity:
        print(f"🏆 최적 밝기: {best_intensity.upper()} (Cohen's d = {best_separation:.3f})")
        if best_separation >= 1.5:
            print("→ 매우 큰 효과 크기 (✓ 진단 적합)")
        elif best_separation >= 0.8:
            print("→ 큰 효과 크기 (✓ 진단 가능)")
        elif best_separation >= 0.5:
            print("→ 중간 효과 크기 (△ 추가 검토)")
        else:
            print("→ 작은 효과 크기 (✗ 조명 개선 필요)")

    # 저장
    result = {
        "base_dir": str(base),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "intensities": all_results,
        "best_intensity": best_intensity,
        "best_separation": float(best_separation) if best_intensity else None,
        "fails": [{"file": f[0], "intensity": f[1], "category": f[2], "reason": f[3]} for f in all_fails],
        "preview_dir": str(preview_root),
    }
    out_json = analysis_root / "summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n[저장 완료] {out_json}")
    print(f"[프리뷰 폴더] {preview_root}")

if __name__ == "__main__":
    main()

# python pilot_4.py