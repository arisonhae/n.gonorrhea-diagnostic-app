# step3_threshold_analysis.py
# Solo 데이터만 사용. JSON 불러오지 않고 모델로 직접 ROI 검출 → 밝기 계산(G-p95) → 음성 기준선(cutoff)으로 양/음성 판정.
# 고정: conf=0.70, method='G', metric='p95' (Δ% 룰 제거)

import argparse, csv, json
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats

# ---- YOLO (ultralytics) ----
try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("Ultralytics YOLO가 필요합니다. pip install ultralytics") from e

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def list_images(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]

def is_in(path: str, root: str) -> bool:
    return str(root).lower().replace("\\","/") in str(path).lower().replace("\\","/")

def solo_label_from_path(path: str) -> str|None:
    low = str(path).lower()
    if "/pos/" in low or low.endswith("/pos") or "\\pos\\" in str(path): return "pos"
    if "/neg/" in low or low.endswith("/neg") or "\\neg\\" in str(path): return "neg"
    return None

# --- 채널/메트릭 고정: method='G', metric='p95' ---
def g_p95_intensity(crop_bgr: np.ndarray) -> float:
    G = crop_bgr[:,:,1].astype(np.float32)
    return float(np.percentile(G, 95))

def safe_crop(img, xyxy):
    x1,y1,x2,y2 = [int(v) for v in xyxy]
    H,W = img.shape[:2]
    x1=max(0,x1); y1=max(0,y1); x2=min(W-1,x2); y2=min(H-1,y2)
    if x2<=x1 or y2<=y1: return None
    return img[y1:y2, x1:x2]

def qq_plot(values, out_png: Path, title: str):
    if len(values) < 3: return
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    stats.probplot(values, dist="norm", plot=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200); plt.close(fig)

# ---------- 시각화 (굵은 라인 + conf 표기) ----------
def draw_box_with_label(img, xyxy, color, label: str, thickness: int = 4):
    """
    박스와 라벨(Conf 포함)을 함께 표시.
    thickness 기본값 4로 강화, 라벨 배경 박스 + 흰색 텍스트.
    """
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(img, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser(description="Step3 threshold analysis (solo only; ROI detection inside)")
    ap.add_argument("--weights", type=str, required=True, help=r'YOLO weights (e.g., C:\...\models\weights.pt)')
    ap.add_argument("--solo_train_root", type=str, default=r"C:\n.gonorrhea_diagnostic_app\dataset\solo\train")
    ap.add_argument("--solo_test_root",  type=str, default=r"C:\n.gonorrhea_diagnostic_app\dataset\solo\test")
    ap.add_argument("--out_dir", type=str, default=r"C:\n.gonorrhea_diagnostic_app\analysis_output\step3")
    # conf는 아래에서 0.70으로 강제
    ap.add_argument("--conf", type=float, default=0.70)
    ap.add_argument("--iou",  type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--class_tube_name", type=str, default="tube")
    ap.add_argument("--class_roi_name",  type=str, default="roi")
    ap.add_argument("--save_viz", action="store_true", help="검출 오버레이 저장")
    args = ap.parse_args()

    # ---- 고정 조건 강제 ----
    args.conf = 0.70  # 반드시 0.70
    METHOD = "G"; METRIC = "p95"

    out_dir = ensure_dir(Path(args.out_dir))
    viz_dir = ensure_dir(out_dir/"viz") if args.save_viz else None

    # ---- 모델 로드 ----
    model = YOLO(args.weights)
    names = model.model.names if hasattr(model.model, "names") else model.names
    try:
        tube_cls = [k for k,v in names.items() if str(v).lower()==args.class_tube_name.lower()][0]
        roi_cls  = [k for k,v in names.items() if str(v).lower()==args.class_roi_name.lower()][0]
    except Exception:
        raise RuntimeError(f"클래스 이름을 찾을 수 없습니다. names={names}")

    # ---- 이미지 목록 ----
    train_root = Path(args.solo_train_root)
    test_root  = Path(args.solo_test_root)
    imgs = list_images(train_root) + list_images(test_root)
    if len(imgs)==0:
        raise RuntimeError("solo/train, solo/test에서 이미지가 발견되지 않았습니다.")

    # ---- 추론 & ROI intensity 수집 ----
    rows = []   # image_id, image_path, split, label, I (G-p95 per image; max over ROIs)
    missing = []
    for ip in imgs:
        label = solo_label_from_path(str(ip))
        if label not in {"pos","neg"}: continue
        split = "train" if is_in(ip, train_root) else ("test" if is_in(ip, test_root) else None)
        if split is None: continue

        img = cv2.imread(str(ip))
        if img is None:
            missing.append(f"IMREAD_FAIL\t{ip}"); continue

        res = model.predict(source=str(ip), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                            device=args.device, verbose=False)[0]

        roi_boxes, tube_boxes = [], []
        for b, c, confv in zip(res.boxes.xyxy.cpu().numpy(),
                               res.boxes.cls.cpu().numpy(),
                               res.boxes.conf.cpu().numpy()):
            if int(c) == roi_cls:   roi_boxes.append((b, float(confv)))
            elif int(c) == tube_cls: tube_boxes.append((b, float(confv)))

        if len(roi_boxes)==0:  # ROI 없음 → 제외
            continue

        # 모든 ROI에서 G-p95 계산 → 그 중 최대값을 이미지 대표값 I
        vals = []
        for (xyxy, confv) in roi_boxes:
            crop = safe_crop(img, xyxy)
            if crop is None: continue
            vals.append(g_p95_intensity(crop))
        if len(vals)==0: continue

        I = float(np.max(vals))
        rows.append({
            "image_id": ip.stem,
            "image_path": str(ip),
            "split": split,
            "label": label,
            "I": f"{I:.6f}"
        })

        if viz_dir is not None:
            draw = img.copy()
            for (xyxy, confv) in tube_boxes:
                draw_box_with_label(draw, xyxy, (0, 255, 0), f"tube ({confv:.2f})", 4)
            for (xyxy, confv) in roi_boxes:
                draw_box_with_label(draw, xyxy, (255, 0, 255), f"roi ({confv:.2f})", 4)
            cv2.imwrite(str(viz_dir/f"{ip.stem}.jpg"), draw)

    if missing:
        (out_dir/"missing_images.txt").write_text("\n".join(missing), encoding="utf-8")

    values_csv = out_dir/"solo_values.csv"
    with open(values_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["image_id","image_path","split","label","I"])
        w.writeheader(); [w.writerow(r) for r in rows]
    print(f"[save] {values_csv}")

    # ---- Split별 배열 ----
    train_neg = [float(r["I"]) for r in rows if r["split"]=="train" and r["label"]=="neg"]
    test_neg  = [float(r["I"]) for r in rows if r["split"]=="test"  and r["label"]=="neg"]
    test_pos  = [float(r["I"]) for r in rows if r["split"]=="test"  and r["label"]=="pos"]

    if len(train_neg)<3:
        print("[warn] train neg 샘플이 적거나 ROI가 안 잡힌 이미지가 많습니다.")

    # ---- Negative baseline (train/neg) → cutoff 산정 ----
    if len(train_neg)>=3:
        sh_W, sh_p = stats.shapiro(train_neg)
    else:
        sh_W, sh_p = (np.nan, np.nan)

    mean_v   = float(np.mean(train_neg)) if len(train_neg)>0 else np.nan
    std_v    = float(np.std(train_neg, ddof=1)) if len(train_neg)>1 else np.nan
    median_v = float(np.median(train_neg)) if len(train_neg)>0 else np.nan
    skew_v   = float(stats.skew(train_neg, bias=False)) if len(train_neg)>2 else np.nan
    kurt_v   = float(stats.kurtosis(train_neg, bias=False)) if len(train_neg)>3 else np.nan
    p997     = float(np.percentile(train_neg, 99.7)) if len(train_neg)>0 else np.nan

    normal = (sh_p>0.05) if not np.isnan(sh_p) else False
    cutoff = (mean_v + 3.0*std_v) if normal else p997

    qq_plot(train_neg, out_png=out_dir/"neg_baseline_qq.png", title="Q-Q plot (train NEG)")

    # ---- Test 평가: cutoff 기반 단일-임계 분류 ----
    def eval_cutoff(pos_vals, neg_vals, cutoff):
        TP = sum(1 for v in pos_vals if v >= cutoff)
        FN = sum(1 for v in pos_vals if v <  cutoff)
        FP = sum(1 for v in neg_vals if v >= cutoff)
        TN = sum(1 for v in neg_vals if v <  cutoff)
        total = max(1, (len(pos_vals)+len(neg_vals)))
        acc = (TP+TN)/total
        prec = TP/max(1, TP+FP)
        rec  = TP/max(1, TP+FN)
        f1   = 2*prec*rec/max(1e-12, (prec+rec))
        return dict(TP=TP,FN=FN,FP=FP,TN=TN,ACC=acc,PREC=prec,RECALL=rec,F1=f1)

    test_eval = eval_cutoff(test_pos, test_neg, cutoff)

    # ---- 저장물 ----
    neg_stats = {
        "n_train_neg": len(train_neg),
        "shapiro_W": None if np.isnan(sh_W) else float(sh_W),
        "shapiro_p": None if np.isnan(sh_p) else float(sh_p),
        "is_normal_by_p>0.05": bool(normal),
        "mean": mean_v, "std": std_v, "median": median_v, "skew": skew_v, "kurtosis": kurt_v,
        "p99_7": p997, "cutoff": cutoff,
        "best_channel_fixed": {"method": METHOD, "metric": METRIC}
    }
    (out_dir/"neg_baseline_stats.json").write_text(json.dumps(neg_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out_dir/"neg_train_values.csv","w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["I_neg_train"]); [w.writerow([f"{v:.6f}"]) for v in train_neg]
    print("[save] neg_baseline_stats.json / neg_baseline_qq.png")

    summary = {
        "settings_fixed": {"conf": 0.70, "method": METHOD, "metric": METRIC},
        "negative_baseline": neg_stats,
        "test_eval_negative_cut": test_eval,
        "files": {
            "solo_values_csv": str(out_dir/"solo_values.csv"),
            "neg_train_values_csv": str(out_dir/"neg_train_values.csv"),
            "neg_baseline_stats_json": str(out_dir/"neg_baseline_stats.json"),
            "neg_baseline_qq_png": str(out_dir/"neg_baseline_qq.png")
        }
    }
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # ---- 콘솔 출력 ----
    print("[done] summary.json saved.")
    print(f"- cutoff (neg baseline) = {cutoff:.4f}  (normal={neg_stats['is_normal_by_p>0.05']})")
    print(f"- Test: ACC={test_eval['ACC']:.3f}, PREC={test_eval['PREC']:.3f}, RECALL={test_eval['RECALL']:.3f}, F1={test_eval['F1']:.3f}")
    print(f"- Fixed settings: conf=0.70, method={METHOD}, metric={METRIC}")

if __name__=="__main__":
    main()
