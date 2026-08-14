# negneg_false_positive_check.py
# neg-neg pair 이미지(위: NC, 아래: Sample)에서
# step4와 완전히 동일한 방식(YOLO + conf 고정 + G-p95)으로
# I_nc, I_sample, ratio를 계산하여 CSV로 저장하는 스크립트.
#
# 실행 예시:
#   python C:\n.gonorrhea_diagnostic_app\analysis_code\step_1117.py ^
#       --weights "C:\n.gonorrhea_diagnostic_app\weights\best.pt"
#
# 기본 설정:
#   - 이미지 폴더: C:\n.gonorrhea_diagnostic_app\dataset\pair\neg_neg
#   - 출력 폴더:   C:\n.gonorrhea_diagnostic_app\analysis_output
#
# 결과 파일:
#   C:\n.gonorrhea_diagnostic_app\analysis_output\neg_neg_pair_analysis.csv
#
# CSV 컬럼은 step4의 pair_analysis와 동일하게 맞춰 두었기 때문에
# viewer 코드에서 거의 그대로 재사용할 수 있음.

import argparse
from pathlib import Path
import numpy as np
import cv2
import csv

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Ultralytics YOLO 패키지가 필요합니다.  (pip install ultralytics)") from e

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def log(msg: str):
    print(str(msg), flush=True)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]


def safe_crop(img, xyxy):
    """step4와 동일한 방식으로 bbox를 잘라내는 함수."""
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    H, W = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W - 1, x2)
    y2 = min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def center_y(b):
    """ROI 박스의 세로(중심 y 좌표)를 반환 → 위/아래 정렬용. (step4와 동일)"""
    x1, y1, x2, y2 = [float(v) for v in b]
    return 0.5 * (y1 + y2)


def g_p95_intensity(crop_bgr: np.ndarray) -> float:
    """
    G-p95 방식:
    - crop의 G 채널을 float32로 변환
    - 전체 픽셀을 1차원으로 펴고
    - 상위 5% 밝기값(95 percentile)을 대표값으로 사용
    """
    if crop_bgr is None:
        return np.nan
    g = crop_bgr[:, :, 1].astype(np.float32)
    flat = g.reshape(-1)
    if flat.size == 0:
        return np.nan
    return float(np.percentile(flat, 95))


def draw_boxes(image, tubes, rois, save_path: Path):
    """디버그용: 튜브 박스(초록), ROI 박스(분홍)를 그려 저장."""
    draw = image.copy()
    for xyxy in tubes:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for xyxy in rois:
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(draw, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.imwrite(str(save_path), draw)


def build_argparser():
    ap = argparse.ArgumentParser(description="neg-neg pair 이미지 분석 (step4 방식 그대로 적용)")
    # 경로
    ap.add_argument(
        "--weights",
        required=True,
        help="YOLO 가중치(.pt) 경로 (step4에서 사용하던 것과 동일한 파일을 넣어야 함)",
    )
    ap.add_argument(
        "--neg_neg_root",
        default=r"C:\n.gonorrhea_diagnostic_app\dataset\pair\neg_neg",
        help="neg-neg pair 이미지 폴더 루트",
    )
    ap.add_argument(
        "--out_dir",
        default=r"C:\n.gonorrhea_diagnostic_app\analysis_output",
        help="분석 결과(output CSV, 시각화 이미지)를 저장할 폴더",
    )
    # 모델/추론 설정: step4와 동일 값 사용
    ap.add_argument("--conf", type=float, default=0.70, help="YOLO confidence threshold")
    ap.add_argument("--iou", type=float, default=0.50, help="YOLO NMS IoU")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO 입력 이미지 크기")
    ap.add_argument("--device", type=str, default="cpu", help="YOLO device (cpu 또는 cuda)")
    # 기타 옵션
    ap.add_argument(
        "--save_viz",
        type=str,
        default="on",
        choices=["on", "off"],
        help="ROI 박스가 그려진 디버그 이미지를 저장할지 여부",
    )
    ap.add_argument(
        "--max_n",
        type=int,
        default=0,
        help="0이면 전체, 0보다 크면 앞에서부터 해당 개수만 분석 (디버그용)",
    )
    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    neg_root = Path(args.neg_neg_root)
    out_dir = ensure_dir(Path(args.out_dir))
    if args.save_viz == "on":
        viz_dir = ensure_dir(out_dir / "viz_neg_neg")
    else:
        viz_dir = None

    log("========== step_1117: neg-neg pair 분석 시작 ==========")
    log(f"[PATH] neg-neg root : {neg_root}")
    log(f"[PATH] out_dir      : {out_dir}")

    # ----- YOLO 로드 (step4와 동일 방식) -----
    log("[MODEL] Loading YOLO weights...")
    model = YOLO(args.weights)
    names = model.model.names if hasattr(model, "model") and hasattr(model.model, "names") else model.names

    try:
        tube_cls = [k for k, v in names.items() if str(v).lower() == "tube"][0]
        roi_cls = [k for k, v in names.items() if str(v).lower() == "roi"][0]
    except Exception:
        raise RuntimeError(f"'tube' / 'roi' 클래스를 names에서 찾지 못했습니다. names={names}")

    log(f"[MODEL] Classes: tube={tube_cls}, roi={roi_cls}")

    def infer_one(path: Path):
        """단일 이미지에 대해 튜브/ROI bbox를 검출 (step4 infer_one과 동일 구조)."""
        res = model.predict(
            source=str(path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]
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

    # ----- neg-neg pair 이미지 목록 수집 -----
    imgs = list_images(neg_root)
    imgs = sorted(imgs)
    if args.max_n > 0:
        imgs = imgs[: args.max_n]
    log(f"[DATA] neg-neg images: {len(imgs)} 장")

    rows = []
    I_nc_list = []
    I_sm_list = []

    for i, ip in enumerate(imgs, 1):
        if (i == 1) or (i % 5 == 0) or (i == len(imgs)):
            log(f"[NEG-NEG] {i}/{len(imgs)} : {ip.name}")

        img, tubes, rois = infer_one(ip)
        if img is None:
            rows.append(
                {
                    "image_path": str(ip),
                    "I_nc": "",
                    "I_sample": "",
                    "delta_pct": "",
                    "ratio": "",
                    "rule": "",
                    "pred_upper": "",
                    "pred_lower": "",
                    "note": "IMREAD_FAIL",
                }
            )
            continue

        rois_sorted = sorted(rois, key=center_y)
        if len(rois_sorted) < 2:
            note = "ROI_PARTIAL" if len(rois_sorted) == 1 else "ROI_NONE"
            rows.append(
                {
                    "image_path": str(ip),
                    "I_nc": "",
                    "I_sample": "",
                    "delta_pct": "",
                    "ratio": "",
                    "rule": "",
                    "pred_upper": "",
                    "pred_lower": "",
                    "note": note,
                }
            )
            if viz_dir is not None:
                draw_boxes(img, tubes, rois, viz_dir / f"negneg__{ip.stem}.jpg")
            continue

        # 위/아래 ROI (위 = NC, 아래 = Sample) — step4 pair와 동일
        ur, lr = rois_sorted[0], rois_sorted[1]
        crop_u = safe_crop(img, ur)
        crop_l = safe_crop(img, lr)
        Iu = g_p95_intensity(crop_u)
        Il = g_p95_intensity(crop_l)

        # Δ% 및 ratio 계산 (step4와 동일 공식)
        m = (Iu + Il) / 2.0
        delta = abs(Il - Iu) / m * 100.0 if m > 0 else np.nan
        ratio = (Il / Iu) if (Iu > 0) else np.nan

        if np.isfinite(Iu):
            I_nc_list.append(Iu)
        if np.isfinite(Il):
            I_sm_list.append(Il)

        rows.append(
            {
                "image_path": str(ip),
                "I_nc": f"{Iu:.6f}" if np.isfinite(Iu) else "",
                "I_sample": f"{Il:.6f}" if np.isfinite(Il) else "",
                "delta_pct": f"{delta:.6f}" if np.isfinite(delta) else "",
                "ratio": f"{ratio:.6f}" if np.isfinite(ratio) else "",
                # neg-neg 분석용이므로 rule / pred_* 는 비워두고 viewer에서 ratio 기준(1.148)만 확인
                "rule": "",
                "pred_upper": "",
                "pred_lower": "",
                "note": "",
            }
        )

        if viz_dir is not None:
            # ROI 2개만 강조해서 저장 (step4 pair와 유사)
            draw_boxes(img, tubes, [ur, lr], viz_dir / f"negneg__{ip.stem}.jpg")

    # ----- CSV 저장 (step4 pair_analysis와 같은 컬럼 구조) -----
    csv_path = out_dir / "neg_neg_pair_analysis.csv"
    fieldnames = [
        "image_path",
        "I_nc",
        "I_sample",
        "delta_pct",
        "ratio",
        "rule",
        "pred_upper",
        "pred_lower",
        "note",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    log(f"[SAVE] neg-neg pair 분석 결과 CSV: {csv_path}")

    if len(I_nc_list) > 0 and len(I_sm_list) > 0:
        med_nc = float(np.median(I_nc_list))
        med_sm = float(np.median(I_sm_list))
        log(f"[STATS] median I_nc = {med_nc:.3f}  ·  median I_sample = {med_sm:.3f}")
    log("========== step_1117: 완료 ==========")


if __name__ == "__main__":
    main()
