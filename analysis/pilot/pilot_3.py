# fluorescence_test_3.py
# 튜브 회전 각도별 형광값 안정성 테스트
# - test_1과 동일한 ROI 추출 방식 사용
# - positive/negative 각각 독립적으로 회전 안정성 분석
# - r_0 기준 편차 계산 및 CV(변동계수) 산출

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

from paths import PILOT_3_ROTATION, OUT_PILOT
BASE_DIR = str(PILOT_3_ROTATION)
# =========================================================

# ========== 설정 ==========
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ========== 공통 유틸 ==========
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
    """크기 제한 내 비율 유지 리사이즈"""
    h, w = bgr.shape[:2]
    if max(h, w) <= max_side:
        return bgr
    s = max_side / max(h, w)
    return cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


# ========== 신호맵 & ROI 검출 (test_1 동일) ==========
def auto_signal_map(bgr):
    """파장 무관 자동 신호맵 (Blue/White 모두 커버, NaN 방어)"""
    f = bgr.astype(np.float32)
    B, G, R = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    eps = 1e-6

    # 후보1: Blue 배경
    alpha = np.median(G) / (np.median(R) + eps)
    cand1 = G - alpha * R - 0.15 * B

    # 후보2: White 배경
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    bg = cv2.medianBlur(Y.astype(np.uint8), 31).astype(np.float32)
    cand2 = Y - bg

    # 후보3: G 채널 자체
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

    return best_map if best_map is not None else normalize_u8(G)


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


# ========== 단일 이미지 분석 ==========
def analyze_one_image(img_path: Path, save_preview_dir: Path):
    print(f"[DEBUG] Processing: {img_path.name}")
    bgr0 = imread_safe(img_path)
    if bgr0 is None:
        raise RuntimeError("이미지 로딩 실패")
    bgr = resize_keep(bgr0, 1200)

    sig = auto_signal_map(bgr)
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

    # 오버레이 저장
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


# ========== 회전 파일 분류 ==========
def parse_rotation_files(folder_path: Path):
    """r_0_1.jpg → angle=0, files=[r_0_1.jpg, r_0_2.jpg]; r_45.png → angle=45"""
    angle_dict: dict[int, list[Path]] = {}
    for ext in IMG_EXTS:
        for f in folder_path.glob(f"r_*{ext}"):
            parts = f.stem.split("_")
            if len(parts) < 2 or not parts[1].lstrip("-").isdigit():
                continue
            angle = int(parts[1])
            angle_dict.setdefault(angle, []).append(f)
    for k in list(angle_dict.keys()):
        angle_dict[k] = sorted(angle_dict[k], key=lambda p: p.name)
    return dict(sorted(angle_dict.items(), key=lambda x: x[0]))


# ========== 회전 분석 ==========
def run_rotation_batch(folder_path: Path, preview_root: Path, tag: str):
    """각도별로 이미지 분석 후 평균 계산"""
    angle_files = parse_rotation_files(folder_path)
    results = {}
    fails = []
    prev_dir = preview_root / tag
    prev_dir.mkdir(parents=True, exist_ok=True)

    for angle, files in angle_files.items():
        print(f"\n[{tag}] 각도 {angle}도 분석 중... ({len(files)}장)")
        angle_metrics = []
        for f in files:
            try:
                m = analyze_one_image(f, prev_dir)
                angle_metrics.append(m)
            except Exception as e:
                fails.append((f.name, str(e)))

        if angle_metrics:
            results[angle] = {
                "sig_mean": float(np.mean([m["sig_mean"] for m in angle_metrics])),
                "bg_mean": float(np.mean([m["bg_mean"] for m in angle_metrics])),
                "snr": float(np.median([m["snr"] for m in angle_metrics])),
                "sat": float(np.median([m["sat"] for m in angle_metrics])),
                "samples": len(angle_metrics),
            }
    return results, fails


# ========== 통계 메트릭 ==========
def calc_stability_metrics(results):
    """CV, 평균, 표준편차 계산"""
    if not results:
        return None
    sig_means = [r["sig_mean"] for r in results.values()]
    mean_val = float(np.mean(sig_means))
    std_val = float(np.std(sig_means, ddof=1)) if len(sig_means) > 1 else 0.0
    eps = 1e-9
    cv = (std_val / (abs(mean_val) + eps)) * 100.0
    return {
        "CV": cv,
        "mean": mean_val,
        "std": std_val,
        "max": float(np.max(sig_means)),
        "min": float(np.min(sig_means)),
        "robust": cv < 10.0,
    }


def calc_deviation_from_baseline(results, baseline_angle=0):
    """r_0 기준 편차 계산 (%)"""
    if baseline_angle not in results:
        return {}
    baseline = float(results[baseline_angle]["sig_mean"])
    eps = 1e-9
    deviations = {}
    for angle, r in results.items():
        dev = ((float(r["sig_mean"]) - baseline) / (baseline + eps)) * 100.0
        deviations[angle] = float(dev)
    return deviations


# ========== 메인 ==========
def main():
    base = Path(BASE_DIR)
    pos_dir = base / "positive"
    neg_dir = base / "negative"
    print("=== 경로 점검 ===")
    for d in (pos_dir, neg_dir):
        print(f"{d}: {'OK' if d.is_dir() else 'MISSING'}")
    print()

    analysis_root = OUT_PILOT / "pilot_3_rotation"
    preview_root = analysis_root / "previews"
    analysis_root.mkdir(exist_ok=True, parents=True)
    preview_root.mkdir(exist_ok=True, parents=True)

    print("=== Positive 회전 분석 시작 ===")
    pos_results, pos_fails = run_rotation_batch(pos_dir, preview_root, "positive")
    print("\n=== Negative 회전 분석 시작 ===")
    neg_results, neg_fails = run_rotation_batch(neg_dir, preview_root, "negative")

    for name, why in (pos_fails + neg_fails):
        print(f"[FAIL] {name} -> {why}")

    pos_stability = calc_stability_metrics(pos_results)
    neg_stability = calc_stability_metrics(neg_results)
    pos_deviation = calc_deviation_from_baseline(pos_results, 0)
    neg_deviation = calc_deviation_from_baseline(neg_results, 0)

    common_angles = sorted(set(pos_results.keys()) & set(neg_results.keys()))
    result = {
        "base_dir": str(base),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "positive": {
            "angles": pos_results,
            "stability": pos_stability,
            "deviation_from_r0": pos_deviation,
        },
        "negative": {
            "angles": neg_results,
            "stability": neg_stability,
            "deviation_from_r0": neg_deviation,
        },
        "common_angles": common_angles,
        "conclusion": {
            "positive_robust": pos_stability["robust"] if pos_stability else False,
            "negative_robust": neg_stability["robust"] if neg_stability else False,
            "recommendation": "회전 각도와 무관하게 안정적 측정 가능"
            if (
                pos_stability
                and neg_stability
                and pos_stability["robust"]
                and neg_stability["robust"]
            )
            else "회전 각도에 따른 형광값 변동 주의 필요",
        },
        "fails": {"positive": pos_fails, "negative": neg_fails},
        "preview_dir": str(preview_root),
    }

    out_json = analysis_root / "summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n=== 완료 ===")
    print(f"[saved] {out_json}")
    print(f"[preview dir] {preview_root}")
    if pos_stability:
        print(
            f"Positive CV: {pos_stability['CV']:.2f}% ({'안정적' if pos_stability['robust'] else '주의'})"
        )
    if neg_stability:
        print(
            f"Negative CV: {neg_stability['CV']:.2f}% ({'안정적' if neg_stability['robust'] else '주의'})"
        )


if __name__ == "__main__":
    main()