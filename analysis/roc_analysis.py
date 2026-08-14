# roc_analysis.py
# YOLO ROI 검출 + 형광(G-p95) 계산 → ROC curve 계산 및 저장 + ROI 시각화 저장
# 실행 예:
#   python "C:\n.gonorrhea_diagnostic_app\analysis_code\ROC.py" ^
#       --weights "C:\n.gonorrhea_diagnostic_app\models\new_weights.pt"

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ---------------- 공통 설정 ----------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def log(msg: str):
    print(str(msg), flush=True)


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_images(root: Path):
    files = []
    for ext in IMG_EXTS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def safe_crop(img, xyxy):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    h, w = img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def g_p95_intensity(crop_bgr: np.ndarray) -> float:
    """ROI 내부 G 채널 상위 5% (p95) intensity"""
    if crop_bgr is None:
        return np.nan
    g = crop_bgr[:, :, 1].astype(np.float32)
    flat = g.reshape(-1)
    if flat.size == 0:
        return np.nan
    return float(np.percentile(flat, 95))


def center_y(box):
    return (float(box[1]) + float(box[3])) / 2.0


def solo_label_from_path(p: Path) -> str | None:
    low = str(p).lower()
    if "\\pos\\" in low or "/pos/" in low:
        return "pos"
    if "\\neg\\" in low or "/neg/" in low:
        return "neg"
    return None


# ---------------- ROC 계산 및 저장 ----------------
def save_roc_curve(out_prefix: Path, scores: np.ndarray, labels: np.ndarray, title: str):
    """
    out_prefix: 확장자 제외 경로 (예: out_dir / 'solo_roc_curve')
    scores: 예측 score (연속값)
    labels: 0/1 라벨
    """
    out_csv = out_prefix.with_suffix(".csv")
    out_png = out_prefix.with_suffix(".png")

    if len(scores) == 0 or len(np.unique(labels)) < 2:
        log(f"[WARN] {title} 라벨 종류가 2개 미만이거나 샘플이 0개라 ROC/AUC 계산 불가.")
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["fpr", "tpr", "threshold", "auc", "n_neg", "n_pos"])
        return

    # ROC 계산
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    n_neg = int(np.sum(labels == 0))
    n_pos = int(np.sum(labels == 1))

    # CSV 저장
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fpr", "tpr", "threshold", "auc", "n_neg", "n_pos"])
        for ff, tt, th in zip(fpr, tpr, thresholds):
            writer.writerow([ff, tt, th, roc_auc, n_neg, n_pos])

    log(f"[ROC] 저장 완료: {out_csv} (AUC={roc_auc:.4f}, neg={n_neg}, pos={n_pos})")

    # PNG 저장
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    log(f"[ROC] 곡선 이미지 저장 완료: {out_png}")


# ---------------- 메인 ----------------
def main():
    ap = argparse.ArgumentParser()

    # 입력 경로
    ap.add_argument(
        "--weights",
        required=True,
        help="YOLO 가중치 경로 (예: C:\\n.gonorrhea_diagnostic_app\\models\\new_weights.pt)",
    )
    ap.add_argument(
        "--solo_train_root",
        default=r"C:\n.gonorrhea_diagnostic_app\dataset\solo\train",
    )
    ap.add_argument(
        "--solo_test_root",
        default=r"C:\n.gonorrhea_diagnostic_app\dataset\solo\test",
    )
    ap.add_argument(
        "--pair_negneg_root",
        default=r"C:\n.gonorrhea_diagnostic_app\dataset\pair\neg_neg",
    )
    ap.add_argument(
        "--pair_negpos_root",
        default=r"C:\n.gonorrhea_diagnostic_app\dataset\pair\neg_pos",
    )
    ap.add_argument(
        "--out_dir",
        default=r"C:\n.gonorrhea_diagnostic_app\analysis_output\ROC",
        help="ROC 결과 저장 폴더",
    )

    # YOLO 추론 설정
    ap.add_argument("--conf", type=float, default=0.70)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="", help="빈칸이면 자동 선택")

    ap.add_argument(
        "--max_per_set",
        type=int,
        default=0,
        help="디버깅용: 각 세트에서 앞 N장만 사용 (0이면 전부)",
    )

    # viz 저장 여부
    ap.add_argument(
        "--save_viz",
        action="store_true",
        help="ROI 검출 결과 이미지를 viz 폴더에 저장",
    )

    args = ap.parse_args()
    out_dir = ensure_dir(Path(args.out_dir))

    # viz 폴더 준비
    if args.save_viz:
        viz_root = ensure_dir(out_dir / "viz")
        viz_solo_pos = ensure_dir(viz_root / "solo_pos")
        viz_solo_neg = ensure_dir(viz_root / "solo_neg")
        viz_pair_nn = ensure_dir(viz_root / "pair_negneg")
        viz_pair_np = ensure_dir(viz_root / "pair_negpos")
    else:
        viz_root = viz_solo_pos = viz_solo_neg = viz_pair_nn = viz_pair_np = None

    log("=== ROC.py: G-p95 / Tratio 기반 ROC 계산 ===")
    log(f"[CONFIG] conf={args.conf}, iou={args.iou}, imgsz={args.imgsz}")
    log(f"[OUT_DIR] {out_dir}")

    # ----- YOLO 로드 -----
    log("[MODEL] YOLO 가중치 로드 중...")
    model = YOLO(args.weights)
    names = model.model.names if hasattr(model.model, "names") else model.names

    # tube / roi 클래스 id 찾기
    try:
        tube_cls = [k for k, v in names.items() if str(v).lower() == "tube"][0]
        roi_cls = [k for k, v in names.items() if str(v).lower() == "roi"][0]
    except Exception:
        raise RuntimeError(f"클래스 이름 'tube' 또는 'roi'를 찾지 못했습니다. names={names}")

    log(f"[MODEL] Classes: tube={tube_cls}, roi={roi_cls}")

    def infer_one(path: Path):
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
            return None, [], [], None
        tubes, rois = [], []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            for b, c in zip(boxes, cls_ids):
                if c == tube_cls:
                    tubes.append(b)
                elif c == roi_cls:
                    rois.append(b)
        return img, tubes, rois, res

    # viz 저장 함수
    def save_viz_image(img, tubes, rois, out_path: Path):
        if img is None:
            return
        vis = img.copy()
        # tube 박스는 초록, ROI 박스는 핑크
        for b in tubes:
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for b in rois:
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.imwrite(str(out_path), vis)

    # ================= SOLO: G_p95 절댓값 ROC =================
    solo_scores = []
    solo_labels = []

    train_root = Path(args.solo_train_root)
    test_root = Path(args.solo_test_root)
    solo_imgs = list_images(train_root) + list_images(test_root)
    if args.max_per_set > 0:
        solo_imgs = solo_imgs[: args.max_per_set]

    log(f"[SOLO] images: {len(solo_imgs)}")

    for i, ip in enumerate(solo_imgs, 1):
        if (i == 1) or (i % 10 == 0) or (i == len(solo_imgs)):
            log(f"[SOLO] {i}/{len(solo_imgs)}: {ip}")

        label_str = solo_label_from_path(ip)
        if label_str not in {"neg", "pos"}:
            continue
        label_int = 1 if label_str == "pos" else 0

        img, tubes, rois, res = infer_one(ip)
        if img is None or len(rois) == 0:
            continue

        vals = []
        for r in rois:
            crop = safe_crop(img, r)
            vals.append(g_p95_intensity(crop))
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            continue

        score = float(np.max(vals))  # 이미지 내 가장 밝은 튜브의 G_p95 사용
        solo_scores.append(score)
        solo_labels.append(label_int)

        # viz 저장
        if args.save_viz:
            if label_str == "pos":
                out_viz = viz_solo_pos / ip.name
            else:
                out_viz = viz_solo_neg / ip.name
            save_viz_image(img, tubes, rois, out_viz)

    solo_scores = np.array(solo_scores, dtype=float)
    solo_labels = np.array(solo_labels, dtype=int)
    log(f"[SOLO] 유효 샘플 수: {len(solo_scores)}")

    save_roc_curve(
        out_dir / "solo_roc_curve",
        solo_scores,
        solo_labels,
        title="ROC Curve – Solo (G_p95 intensity)",
    )

    # ================= PAIR: Tratio ROC (위=NC, 아래=Sample) =================
    pair_scores = []
    pair_labels = []

    negneg_root = Path(args.pair_negneg_root)
    negpos_root = Path(args.pair_negpos_root)

    nn_imgs = list_images(negneg_root)
    np_imgs = list_images(negpos_root)

    if args.max_per_set > 0:
        nn_imgs = nn_imgs[: args.max_per_set]
        np_imgs = np_imgs[: args.max_per_set]

    log(f"[PAIR] neg_neg images: {len(nn_imgs)}")
    log(f"[PAIR] neg_pos images: {len(np_imgs)}")

    def process_pair_image(ip: Path, pair_type: str, label_int: int):
        img, tubes, rois, res = infer_one(ip)
        if img is None or len(rois) < 2:
            return

        # y center 기준으로 정렬 → 위: NC, 아래: Sample
        rois_sorted = sorted(rois, key=center_y)
        upper_roi = rois_sorted[0]      # NC
        lower_roi = rois_sorted[-1]     # Sample

        I_nc = g_p95_intensity(safe_crop(img, upper_roi))
        I_sm = g_p95_intensity(safe_crop(img, lower_roi))
        if not (np.isfinite(I_nc) and np.isfinite(I_sm) and I_nc > 0):
            return

        tratio = float(I_sm / I_nc)
        pair_scores.append(tratio)
        pair_labels.append(label_int)

        # viz 저장
        if args.save_viz:
            if pair_type == "neg_neg":
                out_dir_viz = viz_pair_nn
            else:
                out_dir_viz = viz_pair_np
            save_viz_image(img, tubes, rois, out_dir_viz / ip.name)

    # neg_neg → label 0 (음성)
    for i, ip in enumerate(nn_imgs, 1):
        if (i == 1) or (i % 5 == 0) or (i == len(nn_imgs)):
            log(f"[PAIR neg_neg] {i}/{len(nn_imgs)}: {ip}")
        process_pair_image(ip, "neg_neg", 0)

    # neg_pos → label 1 (양성)
    for i, ip in enumerate(np_imgs, 1):
        if (i == 1) or (i % 5 == 0) or (i == len(np_imgs)):
            log(f"[PAIR neg_pos] {i}/{len(np_imgs)}: {ip}")
        process_pair_image(ip, "neg_pos", 1)

    pair_scores = np.array(pair_scores, dtype=float)
    pair_labels = np.array(pair_labels, dtype=int)
    log(f"[PAIR] 유효 샘플 수: {len(pair_scores)} (neg={np.sum(pair_labels==0)}, pos={np.sum(pair_labels==1)})")

    save_roc_curve(
        out_dir / "pair_roc_curve",
        pair_scores,
        pair_labels,
        title="ROC Curve – Pair (Tratio = Sample / NC)",
    )

    log("=== ROC.py 완료: CSV, PNG, (옵션) viz 저장됨 ===")


if __name__ == "__main__":
    main()
