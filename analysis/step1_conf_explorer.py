# -*- coding: utf-8 -*-
"""
step1_conf_explorer.py

목적
    YOLOv8 검출 confidence 임계값을 정한다.
    낮은 conf(기본 0.20)로 넉넉히 검출한 뒤, 임계값을 0.50~0.85 로 올려가며
    tube / roi 가 각각 몇 장에서 살아남는지 집계한다.

    올리면 오검출이 줄지만 진짜 검출도 함께 사라진다.
    그 균형점을 데이터로 확인하는 것이 이 스크립트의 역할이다.

핵심 지표
    tube & roi 동시 통과율 — 파이프라인이 실제로 요구하는 조건이다.
    ROI 가 없으면 형광값을 계산할 수 없고, pair 이미지는 ROI 가 두 개 있어야 한다.
    따라서 pair 폴더에 대해서는 "ROI 2개 이상" 통과율을 따로 집계한다.

출력
    results/step1_conf/
      ├── per_image_summary.csv     이미지별 검출 conf 목록
      ├── threshold_sweep.csv       임계값별 통과 수
      ├── borderline_images.txt     임계값 근처라 판단이 필요한 이미지 목록
      └── viz/                      (--save_viz 지정 시)

실행
    python analysis/step1_conf_explorer.py

    # 애매한 이미지만 오버레이 저장해서 눈으로 확인
    python analysis/step1_conf_explorer.py --save_viz borderline

    # 전부 저장 (용량 주의)
    python analysis/step1_conf_explorer.py --save_viz all

오버레이 저장에 대하여
    검출이 제대로 되는지는 눈으로 봐야 알 수 있으므로 저장 기능은 유지한다.
    다만 원본 크기 PNG 로 전부 저장하면 산출물이 수 GB 에 이르므로,
    JPEG 로 리사이즈하고 기본값은 저장하지 않는 쪽으로 두었다.

수정 이력
    - 원본(newstep1_conf_explorer.py)에 streamlit 코드가 섞여 있어
      import 없이 st.markdown 을 호출, 실행 즉시 NameError 로 중단되었다.
    - 하드코딩 경로를 paths.py 로 이관
    - tube & roi 동시 통과 및 ROI 2개 이상 통과 집계 추가
    - 임계값 근처 이미지를 따로 기록하는 기능 추가

실행 결과 (2026-08, weights.pt, 238장)
    conf 0.70 을 유지하기로 했다.

    0.65 로 낮추면 기기 호환성 검증 이미지가 13/22 에서 16/22 로 늘지만,
    용액이 튄 불량 이미지도 3/20 에서 4/20 으로 통과가 늘어난다.
    진단 시스템에서는 판정 불가가 잘못된 판정보다 안전하므로 0.70 을 택했다.

    통과하지 못한 3장은 ROI 를 2개 모두 검출했으나 두 번째 ROI 의
    confidence 가 0.656~0.678 로 임계값에 근소하게 미달한 경우다.
    3장 모두 Galaxy Note 8 로 촬영한 것이며, 기기별 second_roi_conf 분포는

        Galaxy Note 8  : 0.656 ~ 0.793
        iPhone 13      : 0.801 ~ 0.815
        iPhone 13 Pro  : 0.797 ~ 0.804

    검출 모델에 기기 편향이 있는 것으로 보인다. 임계값을 낮춰 덮기보다
    학습 데이터에 해당 기기 이미지를 보강해 재학습하는 것이 근본 해결이다.

    solo(110장)와 pair/neg_pos(20장)는 conf 0.70 에서 손실이 없다.

qc_test 데이터의 한계
        qc_test 폴더의 이미지는 대부분 단일 튜브로 촬영되었다.
        앱은 pair(튜브 2개)를 전제하므로, 이 이미지들은 촬영 품질과 무관하게
        ROI 가 하나만 검출되어 "판정 불가" 로 처리된다.
        따라서 이 데이터로는 QC 검출 성능을 평가할 수 없다.

        특히 error_blur 에는 pair 이미지가 없다. 흐린 pair 이미지가 검출을
        통과해 잘못된 판정을 내는지 확인할 방법이 없으며,
        검증되지 않은 위험으로 남아 있다.
"""

import argparse
import csv
import sys
from collections import defaultdict
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

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

VIZ_MAX_WIDTH = 1024
VIZ_JPEG_QUALITY = 85

COLOR_TUBE = (0, 255, 0)
COLOR_ROI = (255, 0, 255)
BOX_THICK = 3


# ==================================================================
# 유틸
# ==================================================================
def list_images(root):
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXTS)


def imread_unicode(path: Path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def to_xyxy(b):
    return [int(float(v)) for v in b[:4]]


def draw_label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    overlay = img.copy()
    cv2.rectangle(overlay, (x, max(0, y - th - 8)), (x + tw + 10, y + 2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2, cv2.LINE_AA)


def save_viz(img, tubes, tconf, rois, rconf, out_path: Path):
    draw = img.copy()
    for b, c in zip(tubes, tconf):
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), COLOR_TUBE, BOX_THICK)
        draw_label(draw, f"T {c:.2f}", b[0], b[1], COLOR_TUBE)
    for b, c in zip(rois, rconf):
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), COLOR_ROI, BOX_THICK)
        draw_label(draw, f"R {c:.2f}", b[0], b[1], COLOR_ROI)

    h, w = draw.shape[:2]
    if w > VIZ_MAX_WIDTH:
        s = VIZ_MAX_WIDTH / w
        draw = cv2.resize(draw, (VIZ_MAX_WIDTH, int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path.with_suffix(".jpg")), draw,
                [cv2.IMWRITE_JPEG_QUALITY, VIZ_JPEG_QUALITY])


def group_of(path: Path) -> str:
    """이미지가 속한 데이터 묶음을 경로에서 판정한다."""
    low = str(path).lower().replace("\\", "/")
    if "/qc_test/" in low:
        for t in ("splash", "blur", "light"):
            if t in low:
                return f"qc_{t}"
        return "qc_other"
    if "/pair/neg_neg" in low:
        return "pair_neg_neg"
    if "/pair/" in low:
        return "pair_neg_pos"
    if "/test_all/" in low:
        return "test_all"
    if "/solo/" in low:
        return "solo"
    return "other"


def needs_two_rois(group: str) -> bool:
    """pair 계열은 ROI 가 두 개 잡혀야 판정이 가능하다."""
    return group.startswith("pair") or group == "test_all"


# ==================================================================
# 메인
# ==================================================================
def main():
    ap = argparse.ArgumentParser(description="Step1: 검출 confidence 임계값 스윕")
    ap.add_argument("--weights", default=str(P.WEIGHTS_PATH))
    ap.add_argument("--roots", nargs="*", default=None,
                    help="분석할 폴더들. 생략하면 paths.py 의 기본 세트를 쓴다")
    ap.add_argument("--out_dir", default=str(P.RESULTS_DIR / "step1_conf"))
    ap.add_argument("--conf", type=float, default=0.20,
                    help="검출 하한. 스윕을 위해 낮게 잡는다")
    ap.add_argument("--iou", type=float, default=P.IOU)
    ap.add_argument("--imgsz", type=int, default=P.IMG_SIZE)
    ap.add_argument("--device", default="")
    ap.add_argument("--class_tube_name", default="tube")
    ap.add_argument("--class_roi_name", default="roi")
    ap.add_argument("--save_viz", choices=["off", "borderline", "all"], default="off",
                    help="오버레이 저장. borderline 은 판단이 필요한 이미지만 저장한다")
    ap.add_argument("--border_lo", type=float, default=0.55,
                    help="이 값 이상 border_hi 미만의 검출이 있으면 borderline 으로 본다")
    ap.add_argument("--border_hi", type=float, default=0.75)
    args = ap.parse_args()

    if args.roots:
        roots = [Path(r) for r in args.roots]
    else:
        roots = [P.SOLO_TRAIN, P.SOLO_TEST, P.PAIR_NEGPOS, P.PAIR_NEGNEG,
                 P.QC_SPLASH, P.QC_BLUR, P.QC_LIGHT, P.TEST_ALL]
        roots = [r for r in roots if Path(r).exists()]

    P.check(Path(args.weights))
    out_dir = P.ensure_dir(Path(args.out_dir))
    viz_dir = P.ensure_dir(out_dir / "viz") if args.save_viz != "off" else None

    print("=" * 68)
    print("Step1 · 검출 confidence 임계값 스윕")
    print("=" * 68)
    print(f"  가중치   : {Path(args.weights).name}")
    print(f"  검출 하한 : {args.conf}  (이 값 이상을 모두 수집한 뒤 스윕)")
    print(f"  출력     : {out_dir}")
    print(f"  오버레이 : {args.save_viz}")
    print()

    model = YOLO(str(args.weights))
    names = model.model.names if hasattr(model.model, "names") else model.names
    try:
        tube_id = next(k for k, v in names.items()
                       if str(v).lower() == args.class_tube_name.lower())
        roi_id = next(k for k, v in names.items()
                      if str(v).lower() == args.class_roi_name.lower())
    except StopIteration:
        raise SystemExit(f"tube/roi 클래스를 찾을 수 없습니다. names={names}")
    print(f"[MODEL] classes={names}\n")

    imgs = []
    for r in roots:
        found = list_images(r)
        print(f"  {str(r):60s} {len(found):>4d}장")
        imgs.extend(found)
    imgs = sorted(set(imgs))
    if not imgs:
        raise SystemExit("이미지를 찾지 못했습니다.")
    print(f"\n[INFO] 총 {len(imgs)}장\n")

    # group -> {"total": n, "tube": [...], "roi": [...], "both": [...], "roi2": [...]}
    stats = defaultdict(lambda: {
        "total": 0,
        "tube": np.zeros(len(THRESHOLDS), int),
        "roi": np.zeros(len(THRESHOLDS), int),
        "both": np.zeros(len(THRESHOLDS), int),
        "roi2": np.zeros(len(THRESHOLDS), int),
    })

    rows, borderline = [], []

    for i, ip in enumerate(imgs, 1):
        img = imread_unicode(ip)
        if img is None:
            print(f"  [WARN] 읽기 실패: {ip.name}")
            continue

        r = model.predict(source=img, conf=args.conf, iou=args.iou,
                          imgsz=args.imgsz, device=args.device, verbose=False)[0]

        tubes, tconf, rois, rconf = [], [], [], []
        for b, c, cf in zip(r.boxes.xyxy.cpu().numpy(),
                            r.boxes.cls.cpu().numpy().astype(int),
                            r.boxes.conf.cpu().numpy()):
            if c == tube_id:
                tubes.append(to_xyxy(b)); tconf.append(float(cf))
            elif c == roi_id:
                rois.append(to_xyxy(b)); rconf.append(float(cf))

        g = group_of(ip)
        top_tube = max(tconf) if tconf else 0.0
        top_roi = max(rconf) if rconf else 0.0
        # ROI 두 개가 필요한 경우, 두 번째로 높은 conf 가 관건이다
        roi2_conf = sorted(rconf, reverse=True)[1] if len(rconf) >= 2 else 0.0

        stats[g]["total"] += 1
        for j, thr in enumerate(THRESHOLDS):
            if top_tube >= thr:
                stats[g]["tube"][j] += 1
            if top_roi >= thr:
                stats[g]["roi"][j] += 1
            if top_tube >= thr and top_roi >= thr:
                stats[g]["both"][j] += 1
            if roi2_conf >= thr:
                stats[g]["roi2"][j] += 1

        # 임계값 근처에 걸린 검출이 있으면 눈으로 확인할 대상으로 표시
        all_conf = tconf + rconf
        is_border = any(args.border_lo <= c < args.border_hi for c in all_conf)
        if is_border:
            borderline.append(f"{g}\t{ip.name}\t"
                              f"tube={','.join(f'{c:.2f}' for c in sorted(tconf, reverse=True))}\t"
                              f"roi={','.join(f'{c:.2f}' for c in sorted(rconf, reverse=True))}")

        if viz_dir is not None and (args.save_viz == "all" or is_border):
            save_viz(img, tubes, tconf, rois, rconf, viz_dir / f"{g}__{ip.stem}")

        rows.append({
            "image_id": ip.stem,
            "image_path": str(ip),
            "group": g,
            "n_tube": len(tubes),
            "n_roi": len(rois),
            "top_tube_conf": f"{top_tube:.4f}",
            "top_roi_conf": f"{top_roi:.4f}",
            "second_roi_conf": f"{roi2_conf:.4f}",
            "tube_conf_list": ";".join(f"{c:.4f}" for c in sorted(tconf, reverse=True)),
            "roi_conf_list": ";".join(f"{c:.4f}" for c in sorted(rconf, reverse=True)),
        })

        if i % 40 == 0 or i == len(imgs):
            print(f"  [{i}/{len(imgs)}]")

    # ---------------- 저장 ----------------
    with open(out_dir / "per_image_summary.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    with open(out_dir / "threshold_sweep.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["group", "metric", "total"] + [f"thr_{t:.2f}" for t in THRESHOLDS])
        for g in sorted(stats):
            v = stats[g]
            for key, label in (("tube", "tube_detected"),
                               ("roi", "roi_detected"),
                               ("both", "tube_and_roi"),
                               ("roi2", "two_rois")):
                w.writerow([g, label, v["total"]] + list(v[key]))

    if borderline:
        (out_dir / "borderline_images.txt").write_text(
            "group\tfile\ttube_confs\troi_confs\n" + "\n".join(borderline),
            encoding="utf-8")

    # ---------------- 출력 ----------------
    print()
    print("=" * 68)
    print("임계값별 통과 이미지 수")
    print("=" * 68)

    hdr = "  " + " ".join(f"{t:>5.2f}" for t in THRESHOLDS)
    for g in sorted(stats):
        v = stats[g]
        n = v["total"]
        print(f"\n[{g}]  총 {n}장")
        print(f"  {'':16s}{hdr}")
        metrics = [("tube 검출", "tube"), ("roi 검출", "roi"), ("tube & roi", "both")]
        if needs_two_rois(g):
            metrics.append(("roi 2개 이상", "roi2"))
        for label, key in metrics:
            cells = " ".join(f"{c:>5d}" for c in v[key])
            print(f"  {label:16s}{cells}")

    # 현재 운영값에서의 손실 요약
    idx_070 = THRESHOLDS.index(0.70)
    print()
    print("=" * 68)
    print("현재 운영값 conf=0.70 에서의 상황")
    print("=" * 68)
    for g in sorted(stats):
        v = stats[g]
        n = v["total"]
        key = "roi2" if needs_two_rois(g) else "both"
        label = "ROI 2개" if needs_two_rois(g) else "tube&roi"
        cur = v[key][idx_070]
        best = int(np.max(v[key]))
        best_thr = THRESHOLDS[int(np.argmax(v[key]))]
        loss = best - cur
        msg = f"  {g:16s} {label:9s} {cur:>3d}/{n:<3d}"
        if loss > 0:
            msg += f"   → conf {best_thr:.2f} 이면 {best}/{n} ({loss}장 더 통과)"
        print(msg)

    print()
    if borderline:
        print(f"[확인 필요] 임계값 근처({args.border_lo}~{args.border_hi}) 검출이 있는 "
              f"이미지 {len(borderline)}장 → borderline_images.txt")
        if args.save_viz == "off":
            print("             --save_viz borderline 을 붙이면 해당 이미지만 저장한다.")
    print(f"\n[저장] {out_dir}")


if __name__ == "__main__":
    main()