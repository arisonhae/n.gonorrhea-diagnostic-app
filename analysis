# -*- coding: utf-8 -*-
"""
step1_conf_explorer.py

목적
    YOLOv8 검출 confidence 임계값을 정하기 위한 스윕 분석.
    conf 0.20 으로 넉넉히 검출한 뒤, 임계값을 0.50~0.80 으로 올려가며
    tube / roi 가 각각 몇 장에서 살아남는지 집계한다.
    이 결과로부터 최종 운영값 conf = 0.70 을 선정하였다.

출력
    <out_dir>/per_image_summary.csv   이미지별 검출 conf 기록
    <out_dir>/viz_conf/*.jpg          박스 오버레이 이미지

실행 예
    python step1_conf_explorer.py \
        --weights models/weights.pt \
        --roots dataset/solo/test/neg dataset/solo/test/pos dataset/pair/neg_pos \
        --out_dir analysis_output/step1_conf

수정 이력
    - 원본(newstep1_conf_explorer.py)에 streamlit 코드가 섞여 있어
      import 없이 st.markdown 을 호출, 실행 즉시 NameError 로 중단되었다.
      해당 블록을 제거하고 CLI 전용 스크립트로 정리하였다.
    - 하드코딩 경로를 인자로 분리하였다.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit("ultralytics 가 필요합니다:  pip install ultralytics") from e


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ---- 시각화 스타일 ----
BOX_BOLD = 4
FONT_SCALE = 1.0
FONT_THICK = 2
LABEL_BG_ALPHA = 0.6

COLOR_TUBE = (0, 255, 0)
COLOR_ROI = (0, 0, 255)


def list_images(roots):
    paths = []
    for r in roots:
        r = Path(r)
        if r.is_file() and r.suffix.lower() in IMG_EXTS:
            paths.append(r)
        elif r.is_dir():
            paths.extend(p for p in r.rglob("*") if p.suffix.lower() in IMG_EXTS)
    return sorted(set(paths))


def to_xyxy(b):
    return [int(float(b[0])), int(float(b[1])), int(float(b[2])), int(float(b[3]))]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def draw_label(img, text, x, y, color):
    """반투명 배경 위에 굵은 라벨을 그린다."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICK)
    overlay = img.copy()
    cv2.rectangle(overlay, (x, max(0, y - th - 6)), (x + tw + 8, y + 2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, LABEL_BG_ALPHA, img, 1 - LABEL_BG_ALPHA, 0, img)
    cv2.putText(img, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, (0, 0, 0), FONT_THICK + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, color, FONT_THICK, cv2.LINE_AA)


def tag_from_path(path: Path) -> str:
    """파일 경로에서 QC 조건 태그를 추출한다."""
    low = str(path).lower()
    for t in ("splash", "blur", "light"):
        if t in low:
            return t
    return "all"


def main():
    ap = argparse.ArgumentParser(
        description="YOLO confidence 임계값 스윕 분석 (tube / roi)")
    ap.add_argument("--weights", required=True, help="YOLO 가중치 (.pt)")
    ap.add_argument("--roots", nargs="+", required=True, help="분석할 이미지 폴더들")
    ap.add_argument("--out_dir", required=True, help="결과 저장 폴더")
    ap.add_argument("--conf", type=float, default=0.20,
                    help="검출 하한. 스윕을 위해 낮게 잡는다 (기본 0.20)")
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--class_tube_name", default="tube")
    ap.add_argument("--class_roi_name", default="roi")
    ap.add_argument("--no_viz", action="store_true", help="오버레이 이미지 저장 생략")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    viz_dir = ensure_dir(out_dir / "viz_conf") if not args.no_viz else None
    csv_path = out_dir / "per_image_summary.csv"

    model = YOLO(str(args.weights))
    img_paths = list_images(args.roots)
    if not img_paths:
        raise SystemExit(f"이미지를 찾지 못했습니다: {args.roots}")

    print(f"[INFO] 대상 이미지 {len(img_paths)}장")

    thresholds = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80]
    stats = defaultdict(lambda: {
        "tube": np.zeros(len(thresholds), int),
        "roi": np.zeros(len(thresholds), int),
        "both": np.zeros(len(thresholds), int),
        "total": 0,
    })

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.DictWriter(fp, fieldnames=[
            "image_id", "image_path", "tag",
            "tube_conf_list", "roi_conf_list",
            "top_tube_conf", "top_roi_conf",
        ])
        writer.writeheader()

        for i, img_path in enumerate(img_paths, 1):
            img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] 읽기 실패: {img_path}")
                continue

            r = model.predict(source=img, conf=args.conf, iou=args.iou,
                              imgsz=args.imgsz, device=args.device,
                              verbose=False)[0]

            inv = {v: k for k, v in r.names.items()}
            if args.class_tube_name not in inv or args.class_roi_name not in inv:
                raise SystemExit(f"모델 클래스에 tube/roi 가 없습니다: {r.names}")
            tube_id, roi_id = inv[args.class_tube_name], inv[args.class_roi_name]

            boxes = r.boxes.xyxy.cpu().numpy()
            clses = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            tubes, tubes_conf, rois, rois_conf = [], [], [], []
            for b, c, cf in zip(boxes, clses, confs):
                if c == tube_id:
                    tubes.append(to_xyxy(b)); tubes_conf.append(float(cf))
                elif c == roi_id:
                    rois.append(to_xyxy(b)); rois_conf.append(float(cf))

            top_tube = max(tubes_conf) if tubes_conf else 0.0
            top_roi = max(rois_conf) if rois_conf else 0.0

            if viz_dir is not None:
                canvas = img.copy()
                for tb, tcf in zip(tubes, tubes_conf):
                    cv2.rectangle(canvas, (tb[0], tb[1]), (tb[2], tb[3]),
                                  COLOR_TUBE, BOX_BOLD)
                    draw_label(canvas, f"T{tcf:.2f}", tb[0], tb[1], COLOR_TUBE)
                for rb, rcf in zip(rois, rois_conf):
                    cv2.rectangle(canvas, (rb[0], rb[1]), (rb[2], rb[3]),
                                  COLOR_ROI, BOX_BOLD)
                    draw_label(canvas, f"R{rcf:.2f}", rb[0], rb[1], COLOR_ROI)
                cv2.imwrite(str(viz_dir / f"{img_path.stem}_viz.jpg"), canvas)

            tag = tag_from_path(img_path)
            stats[tag]["total"] += 1
            for j, thr in enumerate(thresholds):
                if top_tube >= thr:
                    stats[tag]["tube"][j] += 1
                if top_roi >= thr:
                    stats[tag]["roi"][j] += 1
                if top_tube >= thr and top_roi >= thr:
                    stats[tag]["both"][j] += 1

            writer.writerow({
                "image_id": img_path.stem,
                "image_path": str(img_path),
                "tag": tag,
                "tube_conf_list": ";".join(f"{c:.4f}" for c in tubes_conf),
                "roi_conf_list": ";".join(f"{c:.4f}" for c in rois_conf),
                "top_tube_conf": f"{top_tube:.4f}",
                "top_roi_conf": f"{top_roi:.4f}",
            })

            if i % 20 == 0 or i == len(img_paths):
                print(f"[{i}/{len(img_paths)}] {img_path.name}")

    # ---- 요약 ----
    print("\n=== 임계값별 통과 이미지 수 ===")
    for tag, v in sorted(stats.items()):
        print(f"\n[{tag}] 총 {v['total']}장")
        for label, arr in (("tube      ", v["tube"]),
                           ("roi       ", v["roi"]),
                           ("tube&roi  ", v["both"])):
            line = "  ".join(f"{thr:.2f}:{cnt:>3d}"
                             for thr, cnt in zip(thresholds, arr))
            print(f"  {label} {line}")

    print(f"\n[저장] {csv_path}")
    if viz_dir is not None:
        print(f"[저장] {viz_dir}")


if __name__ == "__main__":
    main()
