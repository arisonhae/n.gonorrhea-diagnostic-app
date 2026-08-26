# pilot_6.py
# 목적: 이미지에서 두 개의 ROI를 안정적으로 검출하고 Δ% 계산이 가능한지 확인

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# === 경로 설정 ===
# 이 파일에서 실제로 실행되는 것은 이 블록(이중 밝기 검출)뿐이다.
# 아래 103행 이후는 삼중따옴표로 묶여 있어 실행되지 않는다.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from paths import PILOT_6_PAIR, OUT_PILOT

IMG_DIR = PILOT_6_PAIR / "neg_pos"
OUT_DIR = OUT_PILOT / "pilot_6_dualbright"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# === ROI 크기 설정 ===
ROI_W, ROI_H = 400, 250

def detect_two_brightest_regions(gray):
    """가장 밝은 두 영역 중심 검출"""
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)

    thresh_val = np.percentile(norm, 99)
    _, mask = cv2.threshold(norm, thresh_val, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))

    if num_labels <= 2:
        return None

    sorted_idx = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
    top2_idx = sorted_idx[:2]
    centers = [tuple(map(int, centroids[i])) for i in top2_idx]
    centers.sort(key=lambda c: c[1])  # y 기준 (위 → 아래)
    return centers

def extract_roi(img, cx, cy):
    """중심(cx, cy) 기준 ROI 추출"""
    h, w, _ = img.shape
    x1 = max(cx - ROI_W // 2, 0)
    y1 = max(cy - ROI_H // 2, 0)
    x1 = min(x1, w - ROI_W)
    y1 = min(y1, h - ROI_H)
    roi = img[y1:y1+ROI_H, x1:x1+ROI_W]
    return roi, (x1, y1, ROI_W, ROI_H)

summary = []

for img_path in sorted(IMG_DIR.glob("*.jpg")):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[FAIL] {img_path.name} -> 이미지 읽기 실패")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    centers = detect_two_brightest_regions(gray)
    if centers is None:
        print(f"[WARN] {img_path.name} -> 밝은 영역 2개 검출 실패")
        continue

    (x_top, y_top), (x_bottom, y_bottom) = centers
    roi_top, rect_top = extract_roi(img, x_top, y_top)
    roi_bottom, rect_bottom = extract_roi(img, x_bottom, y_bottom)

    mean_top = np.mean(cv2.cvtColor(roi_top, cv2.COLOR_BGR2GRAY))
    mean_bottom = np.mean(cv2.cvtColor(roi_bottom, cv2.COLOR_BGR2GRAY))
    diff_abs = abs(mean_top - mean_bottom)
    diff_pct = diff_abs / np.mean([mean_top, mean_bottom]) * 100

    summary.append({
        "image": img_path.name,
        "mean_top": round(mean_top, 2),
        "mean_bottom": round(mean_bottom, 2),
        "diff_pct": round(diff_pct, 1),
    })

    vis = img.copy()
    cv2.rectangle(vis, (rect_top[0], rect_top[1]),
                  (rect_top[0]+rect_top[2], rect_top[1]+rect_top[3]), (0,255,0), 3)
    cv2.rectangle(vis, (rect_bottom[0], rect_bottom[1]),
                  (rect_bottom[0]+rect_bottom[2], rect_bottom[1]+rect_bottom[3]), (255,0,0), 3)
    cv2.putText(vis, f"Δ%={diff_pct:.1f}", (80, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,255), 3)

    out_path = OUT_DIR / f"{img_path.stem}_dualbright.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"[OK] {img_path.name} | Δ%={diff_pct:.1f}")

# === 요약 저장 ===
out_json = OUT_DIR / "summary_dualbright.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "generated_at": datetime.now().isoformat(),
        "roi_size": (ROI_W, ROI_H),
        "summary": summary
    }, f, indent=2, ensure_ascii=False)

print("\n=== 파일럿 6 요약 ===")
print(f"검출 성공 이미지 수: {len(summary)}")
print(f"Δ 평균: {np.mean([r['diff_pct'] for r in summary]):.1f}%")
print(f"[저장 완료] {out_json}")

# python pilot_6.py
'''
기존 ROI 기반
# pilot_6.py
# 목적: pilotimage_6/neg_pos 폴더의 이미지를 두 개 ROI로 자동 분리되는지 검증
# 산출물: _analysis_6/summary.json, details.csv, previews/*_overlay.jpg, *_signal.jpg

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageOps
import pandas as pd

# ===== 설정 =====
BASE_DIR = r"C:\n.gonorrhea_diagnostic_app\pilotimage_6"
TARGET_SUBDIR = "neg_pos"  # 이번 파일럿은 이 폴더만 처리

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".heic", ".HEIC"}

# (선택) HEIC 지원
HEIC_ENABLED = False
try:
    from pillow_heif import register_heif_opener  # pip install pillow-heif
    register_heif_opener()
    HEIC_ENABLED = True
except Exception:
    pass

# ===== 유틸 =====
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

def resize_keep(bgr, max_side=1400):
    h, w = bgr.shape[:2]
    if max(h, w) <= max_side:
        return bgr
    s = max_side / max(h, w)
    return cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

# ===== 신호맵 =====
def auto_signal_map(bgr):
    """파장 무관 자동 신호맵 (cand1/cand2/cand3 중 score 최고 선택)"""
    f = bgr.astype(np.float32)
    B, G, R = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    eps = 1e-6

    alpha = np.median(G) / (np.median(R) + eps)
    cand1 = G - alpha * R - 0.15 * B  # 형광강조형

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    bgY = cv2.medianBlur(Y.astype(np.uint8), 31).astype(np.float32)
    cand2 = Y - bgY  # 밝기-배경 제거

    cand3 = G  # 단순 그린

    def normalize_u8(S):
        p1, p99 = np.percentile(S, 1), np.percentile(S, 99)
        Sn = (S - p1) / max(p99 - p1, 1e-6)
        Sn = np.clip(Sn, 0, 1)
        return (Sn * 255).astype(np.uint8)

    def safe_score(Su8):
        Su8 = Su8.astype(np.float32)
        top = float(np.percentile(Su8, 99.5))
        med = float(np.median(Su8))
        low_mask = Su8 < np.percentile(Su8, 50)
        low_std = float(np.std(Su8[low_mask])) if np.any(low_mask) else float(np.std(Su8))
        return (top - med) / (low_std + 1e-6)

    best_map, best_sc = None, -1e9
    for S in (cand1, cand2, cand3):
        Su8 = normalize_u8(S)
        sc = safe_score(Su8)
        if sc > best_sc:
            best_sc, best_map = sc, Su8
    return best_map

# ===== ROI & 로컬 배경 =====
def detect_two_roi_masks(sig_u8):
    """상위 blob 2개 ROI, y좌표 기준으로 위→아래 정렬"""
    blur = cv2.GaussianBlur(sig_u8, (5, 5), 1.2)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k, iterations=1)
    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:2]
    masks = []
    for c in cnts:
        m = np.zeros_like(sig_u8, np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        masks.append(m)

    # y 기준 정렬 (위쪽 먼저)
    if len(masks) == 2:
        ys = [np.mean(np.where(m > 0)[0]) if np.any(m > 0) else 1e9 for m in masks]
        order = np.argsort(ys)
        masks = [masks[order[0]], masks[order[1]]]
    return masks

def annulus_bg(sig_u8, mask, gap=3, ring=10):
    """ROI 외곽 환형 배경 평균/표준편차."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*gap+1, 2*gap+1))
    inner = cv2.dilate(mask, k, iterations=1)
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*(gap+ring)+1, 2*(gap+ring)+1))
    outer = cv2.dilate(mask, k2, iterations=1)
    ann = cv2.subtract(outer, inner)
    vals = sig_u8[ann > 0].astype(np.float32)
    if vals.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))

def sat_pct_u8(img_u8, thr=250):
    return float(np.mean(img_u8 >= thr) * 100.0)

def iou_masks(m1, m2):
    inter = np.logical_and(m1 > 0, m2 > 0).sum()
    union = np.logical_or(m1 > 0, m2 > 0).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)

def centroid(mask):
    y, x = np.where(mask > 0)
    if x.size == 0:
        return None
    return (float(np.mean(x)), float(np.mean(y)))  # (cx, cy)

def min_gap_between_masks(m1, m2):
    """두 마스크 사이 최소 거리(픽셀). 간단히 거리변환으로 계산."""
    # 경계 추출
    c1 = cv2.Canny(m1, 0, 1)
    c2 = cv2.Canny(m2, 0, 1)
    if not np.any(c1) or not np.any(c2):
        return None
    dist = cv2.distanceTransform((255 - c2).astype(np.uint8), cv2.DIST_L2, 3)
    d1 = dist[c1 > 0]
    if d1.size == 0:
        return None
    return float(np.min(d1))

# ===== 단일 이미지 분석 =====
def analyze_image(img_path: Path, preview_dir: Path):
    bgr0 = imread_safe(img_path)
    if bgr0 is None:
        raise RuntimeError("이미지 로딩 실패")
    bgr = resize_keep(bgr0, 1400)
    sig = auto_signal_map(bgr)

    masks = detect_two_roi_masks(sig)
    if len(masks) < 2:
        raise RuntimeError(f"ROI가 2개가 아님 (검출={len(masks)})")

    top_mask, bot_mask = masks[0], masks[1]  # 위=NC 후보, 아래=시료 후보
    iou = iou_masks(top_mask, bot_mask)
    gap_px = min_gap_between_masks(top_mask, bot_mask)

    # ROI 통계
    def stat_one(mask):
        roi_vals = sig[mask > 0].astype(np.float32)
        if roi_vals.size < 20:
            raise RuntimeError("ROI 픽셀 부족")
        bg_mean, bg_std = annulus_bg(sig, mask, gap=3, ring=10)
        sig_mean = float(np.mean(roi_vals))
        signal = float(sig_mean - (bg_mean if not np.isnan(bg_mean) else 0.0))
        snr = float((sig_mean - (bg_mean if not np.isnan(bg_mean) else 0.0)) / (bg_std + 1e-6 if not np.isnan(bg_std) else 1.0))
        area = int((mask > 0).sum())
        c = centroid(mask)
        return {
            "sig_mean": sig_mean, "bg_local_mean": bg_mean, "bg_local_std": bg_std,
            "signal": signal, "snr": snr, "area": area, "centroid": c
        }

    top_stat = stat_one(top_mask)
    bot_stat = stat_one(bot_mask)

    # 차이 및 분리 판정 보조지표
    s1, s2 = top_stat["signal"], bot_stat["signal"]
    mean_s = (abs(s1) + abs(s2)) / 2.0 + 1e-6
    delta_pct = float(abs(s1 - s2) / mean_s * 100.0)

    # 간단 분리 기준(참고): IoU<0.02, gap>=5px
    separation_ok = (iou < 0.02) and (gap_px is not None and gap_px >= 5.0)

    # 프리뷰 저장
    preview_dir.mkdir(parents=True, exist_ok=True)
    overlay = bgr.copy()
    for idx, (m, color, tag) in enumerate(
        [(top_mask, (0, 255, 0), "NC? TOP"), (bot_mask, (255, 0, 0), "EXP? BOTTOM")], start=1
    ):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, color, 2)
        c = centroid(m)
        if c:
            cv2.putText(overlay, f"{tag}", (int(c[0])+5, int(c[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    info = f"Δ%={delta_pct:.1f} | IoU={iou:.3f} | gap={gap_px:.1f}px | sep_ok={'Y' if separation_ok else 'N'}"
    cv2.putText(overlay, info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(preview_dir / f"{img_path.stem}_overlay.jpg"), overlay)
    cv2.imwrite(str(preview_dir / f"{img_path.stem}_signal.jpg"), sig)

    return {
        "filename": img_path.name,
        "delta_pct": delta_pct,
        "iou": iou,
        "gap_px": gap_px,
        "separation_ok": bool(separation_ok),
        "sat_pct": sat_pct_u8(sig),
        "top": top_stat,  # 위=NC 후보
        "bottom": bot_stat  # 아래=실험 후보
    }

# ===== 실행 =====
def main():
    base = Path(BASE_DIR)
    target = base / TARGET_SUBDIR
    if not target.exists():
        print(f"ERROR: {target} 폴더가 없습니다.")
        return

    analysis = base / "_analysis_6"
    previews = analysis / "previews"
    analysis.mkdir(parents=True, exist_ok=True)
    previews.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in target.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()])
    if not images:
        print(f"ERROR: {target} 에 처리할 이미지가 없습니다.")
        return

    results = []
    rows = []
    fails = []
    for p in images:
        try:
            r = analyze_image(p, previews)
            results.append(r)

            row = {
                "filename": r["filename"],
                "delta_pct": r["delta_pct"],
                "iou": r["iou"],
                "gap_px": r["gap_px"],
                "separation_ok": r["separation_ok"],
                "sat_pct": r["sat_pct"],
                "top_signal": r["top"]["signal"],
                "bottom_signal": r["bottom"]["signal"],
                "top_snr": r["top"]["snr"],
                "bottom_snr": r["bottom"]["snr"],
                "top_bg": r["top"]["bg_local_mean"],
                "bottom_bg": r["bottom"]["bg_local_mean"],
                "top_area": r["top"]["area"],
                "bottom_area": r["bottom"]["area"],
            }
            rows.append(row)

            print(f"[OK] {p.name} | sep_ok={r['separation_ok']} | Δ%={r['delta_pct']:.1f} | IoU={r['iou']:.3f} | gap={r['gap_px']:.1f}px")
        except Exception as e:
            fails.append({"file": p.name, "reason": str(e)})
            print(f"[FAIL] {p.name} -> {e}")

    # 요약 통계
    sep_flags = [r["separation_ok"] for r in results]
    delta_list = [r["delta_pct"] for r in results]
    iou_list = [r["iou"] for r in results]
    gap_list = [r["gap_px"] for r in results if r["gap_px"] is not None]

    summary = {
        "base_dir": str(base),
        "target_dir": str(target),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_images": len(images),
        "n_success": int(sum(sep_flags)),
        "success_rate": float(np.mean(sep_flags)) * 100.0 if sep_flags else 0.0,
        "delta_pct_mean": float(np.mean(delta_list)) if delta_list else None,
        "delta_pct_std": float(np.std(delta_list)) if delta_list else None,
        "iou_mean": float(np.mean(iou_list)) if iou_list else None,
        "gap_px_mean": float(np.mean(gap_list)) if gap_list else None,
        "details": results,
        "fails": fails,
        "preview_dir": str(previews),
    }

    # 저장
    json_path = analysis / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[저장] {json_path}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(analysis / "details.csv", index=False, encoding="utf-8-sig")
        print(f"[저장] {analysis / 'details.csv'}")

if __name__ == "__main__":
    main()
'''

''''
hsv 기반
# pilot_6_fluoro_mask_qc.py
# 목적: HSV 형광 마스크 + 컨투어로 2개 ROI(TOP/BOTTOM) 안정 검출 + 기본 QC
# 입출력:
#   입력 이미지: C:\n.gonorrhea_diagnostic_app\pilotimage_6\neg_pos\*.jpg
#   결과 이미지 및 요약: C:\n.gonorrhea_diagnostic_app\pilotimage_6\_analysis_maskqc\

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageOps
import pandas as pd

# ===== 경로 =====
IMG_DIR = Path(r"C:\n.gonorrhea_diagnostic_app\pilotimage_6\neg_pos")
OUT_DIR = Path(r"C:\n.gonorrhea_diagnostic_app\pilotimage_6\_analysis_maskqc")
PREV_DIR = OUT_DIR / "previews"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PREV_DIR.mkdir(parents=True, exist_ok=True)

# ===== 파라미터 (필요시 조정) =====
# HSV 형광(노란-황록 계열) 범위
HSV_LOWER = np.array([15, 70, 70])     # H,S,V 최소
HSV_UPPER = np.array([55, 255, 255])   # H,S,V 최대

MIN_AREA = 600          # 컨투어 최소 면적
ASPECT_MIN, ASPECT_MAX = 0.35, 1.2     # bbox 종횡비 W/H 허용 범위
SOLIDITY_MIN = 0.75     # 컨투어 채움도(면적/볼록껍질) 최소

# QC 임계값
BLUR_VAR_MIN = 60.0     # 라플라시안 분산(낮으면 블러)
SAT_RATIO_MAX = 0.12    # 포화 픽셀 비율(ROI 내 250 이상)
LIGHT_LEAK_BG_V = 40    # 배경 V 평균(HSV) 높으면 라이트리크 의심
WALL_TOUCH_RATIO = 0.25 # ROI bbox 측면에 닿은 형광 비율이 이보다 크면 '벽면 묻음' 의심

# ===== 유틸 =====
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

def variance_of_laplacian(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def hsv_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    # 노이즈 정리
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=1)
    return mask, hsv

def contour_sanity(contour):
    area = cv2.contourArea(contour)
    if area < MIN_AREA:
        return False, area, None, None, None
    x, y, w, h = cv2.boundingRect(contour)
    aspect = w / max(h, 1)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull) + 1e-6
    solidity = area / hull_area
    ok = (ASPECT_MIN <= aspect <= ASPECT_MAX) and (solidity >= SOLIDITY_MIN)
    return ok, area, (x, y, w, h), aspect, solidity

def pick_two_rois(mask, bgr):
    """형광 마스크에서 조건을 만족하는 컨투어 중 가장 '튜브스러운' 2개를 y좌표 기준으로 선택"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in cnts:
        ok, area, rect, aspect, solidity = contour_sanity(c)
        if not ok:
            continue
        x, y, w, h = rect
        # ROI 내부 평균 밝기(회색), 포화 비율도 후보 점수에 반영
        roi = bgr[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        sat_ratio = float(np.mean(gray >= 250))
        score = mean - 100*sat_ratio + 0.001*area  # 밝고 적당히 크며 과포화 적은 컨투어 선호
        cy = y + h/2
        cands.append((score, cy, (x, y, w, h), sat_ratio))
    if len(cands) < 2:
        return None
    # 점수 상위 4개 중 y거리로 충분히 떨어진 2개 구성 시도
    cands = sorted(cands, key=lambda t: t[0], reverse=True)[:4]
    # 조합 탐색: y 겹침 적고 세장비 유사한 쌍 선호
    best = None
    for i in range(len(cands)):
        for j in range(i+1, len(cands)):
            _, cy1, r1, _ = cands[i]
            _, cy2, r2, _ = cands[j]
            gap = abs(cy1 - cy2)
            h_mean = 0.5*(r1[3] + r2[3])
            if gap < 0.35*h_mean:  # 세로 중심 간격이 너무 가까우면 제외
                continue
            score_pair = cands[i][0] + cands[j][0] + gap
            if best is None or score_pair > best[0]:
                best = (score_pair, r1, r2)
    if best is None:
        # 그래도 없으면 점수 상위 2개를 사용
        r1 = cands[0][2]
        r2 = cands[1][2]
    else:
        _, r1, r2 = best
    # 위/아래 정렬
    if r1[1] <= r2[1]:
        top, bottom = r1, r2
    else:
        top, bottom = r2, r1
    return top, bottom

def roi_stats_and_qc(bgr, hsv, rect):
    x, y, w, h = rect
    roi = bgr[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 신호: 중앙부 70%만 사용(가장자리 영향 완화)
    ix = int(x + 0.15*w); iy = int(y + 0.15*h)
    iw = int(0.70*w); ih = int(0.70*h)
    core = bgr[iy:iy+ih, ix:ix+iw]
    core_gray = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)

    mean = float(core_gray.mean())
    median = float(np.median(core_gray))
    sat_ratio = float(np.mean(core_gray >= 250.0))

    # 배경: bbox 바깥쪽 링(사각형 확장 - 내부 제외)
    h_img, w_img = bgr.shape[:2]
    expand = int(0.25*max(w, h))
    x0 = max(x - expand, 0); y0 = max(y - expand, 0)
    x1 = min(x + w + expand, w_img); y1 = min(y + h + expand, h_img)
    bg = bgr[y0:y1, x0:x1].copy()
    bg[y - y0:y - y0 + h, x - x0:x - x0 + w] = 0  # 내부 제거
    bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    bg_vals = bg_gray[bg_gray > 0]
    bg_mean = float(bg_vals.mean()) if bg_vals.size else 0.0
    bg_std  = float(bg_vals.std() ) if bg_vals.size else 1.0
    snr = (mean - bg_mean) / (bg_std + 1e-6)

    # QC 지표
    blur_var = variance_of_laplacian(gray)
    # 라이트리크: 전체 배경 HSV의 V 평균(ROI 주변 링)으로 판정
    bg_hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    v_bg = float(bg_hsv[...,2][bg_gray>0].mean()) if bg_vals.size else 0.0
    light_leak = v_bg > LIGHT_LEAK_BG_V

    # 벽면 묻음: ROI 마스크가 bbox 좌/우 가장자리에 닿아있는 비율
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_mask = cv2.inRange(roi_hsv, HSV_LOWER, HSV_UPPER)
    # 좌/우 10% 세로 스트립에서 마스크가 존재하는 비율
    lw = max(int(0.10*w), 1)
    left_strip  = roi_mask[:, :lw]
    right_strip = roi_mask[:, -lw:]
    touch_ratio = float((left_strip>0).mean() + (right_strip>0).mean())/2.0
    wall_smear = touch_ratio > WALL_TOUCH_RATIO

    qc = {
        "blur_var": blur_var,
        "is_blur": blur_var < BLUR_VAR_MIN,
        "sat_ratio": sat_ratio,
        "over_saturation": sat_ratio > SAT_RATIO_MAX,
        "bg_v_mean": v_bg,
        "light_leak": bool(light_leak),
        "wall_touch_ratio": touch_ratio,
        "wall_smear": bool(wall_smear),
    }

    stats = {
        "mean": mean, "median": median, "bg_mean": bg_mean, "bg_std": bg_std,
        "snr": snr, "sat_ratio": sat_ratio
    }
    return stats, qc

def draw_box(img, rect, color, label):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def iou(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1 = max(ax, bx); y1 = max(ay, by)
    x2 = min(ax+aw, bx+bw); y2 = min(ay+ah, by+bh)
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = aw*ah + bw*bh - inter + 1e-6
    return inter/union

# ===== 메인 =====
rows = []
fails = []

for img_path in sorted([p for p in IMG_DIR.iterdir() if p.suffix.lower() in IMG_EXTS]):
    bgr = imread_safe(img_path)
    if bgr is None:
        fails.append({"file": img_path.name, "reason": "imread failed"})
        print(f"[FAIL] {img_path.name} -> 읽기 실패")
        continue

    mask, hsv = hsv_mask(bgr)
    picked = pick_two_rois(mask, bgr)
    if picked is None:
        fails.append({"file": img_path.name, "reason": "no_two_rois"})
        print(f"[FAIL] {img_path.name} -> ROI 2개 검출 실패")
        # 디버그 프리뷰 저장
        cv2.imwrite(str(PREV_DIR / f"{img_path.stem}_mask.jpg"), mask)
        continue

    top_rect, bot_rect = picked
    # 통계/품질
    top_stats, top_qc = roi_stats_and_qc(bgr, hsv, top_rect)
    bot_stats, bot_qc = roi_stats_and_qc(bgr, hsv, bot_rect)

    # 비교 지표
    mean_top, mean_bot = top_stats["mean"], bot_stats["mean"]
    diff_abs = abs(mean_top - mean_bot)
    diff_pct = 100.0 * diff_abs / ( (mean_top + mean_bot)/2.0 + 1e-6 )
    gap_px = abs( (top_rect[1]+top_rect[3]/2) - (bot_rect[1]+bot_rect[3]/2) )
    iou_val = iou(top_rect, bot_rect)

    # 프리뷰
    vis = bgr.copy()
    draw_box(vis, top_rect, (0,255,0), f"TOP {mean_top:.1f}")
    draw_box(vis, bot_rect, (255,0,0), f"BOTTOM {mean_bot:.1f}")
    cv2.putText(vis, f"Delta%={diff_pct:.1f} | IoU={iou_val:.3f} | gap={gap_px:.1f}px",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
    # QC 라벨
    def qc_tag(qc):
        flags = []
        if qc["is_blur"]: flags.append("BLUR")
        if qc["over_saturation"]: flags.append("SAT")
        if qc["light_leak"]: flags.append("LEAK")
        if qc["wall_smear"]: flags.append("WALL")
        return ",".join(flags) if flags else "OK"
    cv2.putText(vis, f"TOP_QC:{qc_tag(top_qc)}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.putText(vis, f"BOT_QC:{qc_tag(bot_qc)}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    cv2.imwrite(str(PREV_DIR / f"{img_path.stem}_overlay.jpg"), vis)
    cv2.imwrite(str(PREV_DIR / f"{img_path.stem}_mask.jpg"), mask)

    row = {
        "file": img_path.name,
        "top_mean": round(mean_top,2),
        "bot_mean": round(mean_bot,2),
        "delta_pct": round(diff_pct,1),
        "gap_px": round(float(gap_px),1),
        "iou": round(float(iou_val),3),

        "top_snr": round(top_stats["snr"],2),
        "bot_snr": round(bot_stats["snr"],2),

        "top_sat": round(top_qc["sat_ratio"],3),
        "bot_sat": round(bot_qc["sat_ratio"],3),

        "top_blur_var": round(top_qc["blur_var"],1),
        "bot_blur_var": round(bot_qc["blur_var"],1),

        "top_light_leak": bool(top_qc["light_leak"]),
        "bot_light_leak": bool(bot_qc["light_leak"]),

        "top_wall": bool(top_qc["wall_smear"]),
        "bot_wall": bool(bot_qc["wall_smear"]),
    }
    rows.append(row)
    print(f"[OK] {img_path.name} | Δ%={row['delta_pct']:.1f} | gap={row['gap_px']:.1f}px | IoU={row['iou']:.3f}")

# 저장
summary = {
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "img_dir": str(IMG_DIR),
    "out_dir": str(OUT_DIR),
    "params": {
        "HSV_LOWER": HSV_LOWER.tolist(),
        "HSV_UPPER": HSV_UPPER.tolist(),
        "MIN_AREA": MIN_AREA, "ASPECT_MIN": ASPECT_MIN, "ASPECT_MAX": ASPECT_MAX,
        "SOLIDITY_MIN": SOLIDITY_MIN, "BLUR_VAR_MIN": BLUR_VAR_MIN,
        "SAT_RATIO_MAX": SAT_RATIO_MAX, "LIGHT_LEAK_BG_V": LIGHT_LEAK_BG_V,
        "WALL_TOUCH_RATIO": WALL_TOUCH_RATIO
    },
    "fails": fails
}

df = pd.DataFrame(rows)
csv_path = OUT_DIR / "results_maskqc.csv"
json_path = OUT_DIR / "summary_maskqc.json"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump({**summary, "rows": rows}, f, ensure_ascii=False, indent=2)

print(f"\n[저장] {csv_path}")
print(f"[저장] {json_path}")
print(f"[프리뷰 폴더] {PREV_DIR}")
'''