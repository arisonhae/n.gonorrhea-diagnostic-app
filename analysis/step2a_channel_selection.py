# step2a_channel_selection.py
# 목적:
#  - JSON을 읽지 않고, YOLOv8으로 직접 탐지(conf=0.70 고정)하여 tube/ROI 매칭
#  - pair: "위=neg, 아래=pos"로 강제 / solo: 폴더명(pos/neg)으로 라벨링
#  - test_all 경로 포함 이미지는 전부 제외
#  - 여러 channel/method(=metric)로 ROI 형광값(mean, p95 등)를 계산
#  - Cohen's d가 가장 큰 (method, metric)을 선정
#  - per-image 값 CSV, 전체 리포트 CSV, best_channel.json 저장
#  - 시각화 이미지(viz) 저장

import argparse
from pathlib import Path
import csv
import json
import cv2
import numpy as np
from math import sqrt

# ----------------- YOLO -----------------
try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("ultralytics가 필요합니다. pip install ultralytics") from e

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ----------------- IO helpers -----------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def list_images(roots):
    paths = []
    for r in roots:
        r = Path(r)
        if r.is_file() and r.suffix.lower() in IMG_EXTS:
            paths.append(r)
        elif r.is_dir():
            for p in r.rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    paths.append(p)
    return sorted(set(paths))

def path_parts_lower(p: str):
    return [s.lower() for s in Path(p).parts]

def is_in_test_all(path: str) -> bool:
    return any("test_all" in part for part in path_parts_lower(path))

def is_pair_image(path: str) -> bool:
    if is_in_test_all(path): return False
    parts = path_parts_lower(path)
    return ("pair" in parts) or any("neg_pos" in s for s in parts)

def is_solo_image(path: str) -> bool:
    if is_in_test_all(path): return False
    parts = path_parts_lower(path)
    return "solo" in parts

def solo_label_from_path(path: str):
    if not is_solo_image(path): return None
    parts = path_parts_lower(path)
    has_pos = any(part == "pos" for part in parts)
    has_neg = any(part == "neg" for part in parts)
    if has_pos and not has_neg: return "pos"
    if has_neg and not has_pos: return "neg"
    return None  # 모호하면 제외

# ----------------- geometry -----------------
def to_xyxy(b):
    return [int(float(b[0])), int(float(b[1])), int(float(b[2])), int(float(b[3]))]

def box_area(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)

def inside(inner, outer):
    x1, y1, x2, y2 = inner
    X1, Y1, X2, Y2 = outer
    return (x1 >= X1 and y1 >= Y1 and x2 <= X2 and y2 <= Y2)

def center_y(b):
    return (b[1] + b[3]) / 2.0

def safe_crop(img, xyxy):
    if img is None or xyxy is None:
        return None
    x1,y1,x2,y2 = [int(v) for v in xyxy]
    H,W = img.shape[:2]
    x1=max(0,x1); y1=max(0,y1); x2=min(W-1,x2); y2=min(H-1,y2)
    if x2<=x1 or y2<=y1: return None
    return img[y1:y2, x1:x2]

# ----------------- intensity methods -----------------
def get_channels(img_bgr):
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    B,G,R = cv2.split(img_bgr)
    H,S,V = cv2.split(hsv)
    return {"B":B, "G":G, "R":R, "HSV_S":S, "HSV_V":V, "GRAY":gray}

def compute_value(method: str, crop_bgr):
    if crop_bgr is None:
        return None
    chs = get_channels(crop_bgr)
    B,G,R = chs["B"], chs["G"], chs["R"]
    V = chs["HSV_V"]; S = chs["HSV_S"]; GRAY = chs["GRAY"]

    eps = 1e-6
    if method == "G":
        M = G.astype(np.float32)
    elif method == "HSV_V":
        M = V.astype(np.float32)
    elif method == "GRAY":
        M = GRAY.astype(np.float32)
    elif method == "HSV_S":
        M = S.astype(np.float32)
    elif method == "G_norm":     # G / (R+G+B) -> [0..255]
        M = G.astype(np.float32) / (R.astype(np.float32)+G.astype(np.float32)+B.astype(np.float32)+eps)
        M *= 255.0
    elif method == "G_ratio":    # G / (R+B) -> normalize to [0..255]
        M = G.astype(np.float32) / (R.astype(np.float32)+B.astype(np.float32)+eps)
        M = np.clip(M, 0, np.percentile(M, 99.9))
        M = (M / (np.max(M)+eps)) * 255.0
    elif method == "ExG":        # 2G - R - B -> [0..255] by contrast stretch
        M = 2.0*G.astype(np.float32) - R.astype(np.float32) - B.astype(np.float32)
        lo, hi = np.percentile(M, 1.0), np.percentile(M, 99.0)
        if hi <= lo: hi = lo + 1.0
        M = np.clip((M - lo) / (hi - lo), 0.0, 1.0) * 255.0
    else:
        return None

    mean_v = float(np.mean(M))
    p95_v  = float(np.percentile(M, 95.0))
    return {"mean_au": mean_v, "p95_au": p95_v}

# ----------------- stats -----------------
def cohens_d(pos_vals, neg_vals):
    pos = np.asarray(pos_vals, dtype=float); neg = np.asarray(neg_vals, dtype=float)
    if len(pos) < 2 or len(neg) < 2: return float("nan")
    m1, m2 = np.mean(pos), np.mean(neg)
    s1, s2 = np.var(pos, ddof=1), np.var(neg, ddof=1)
    n1, n2 = len(pos), len(neg)
    sp = sqrt(((n1-1)*s1 + (n2-1)*s2) / max(1,(n1+n2-2)))
    if sp == 0: return float("nan")
    return (m1 - m2) / sp

# ----------------- viz -----------------
BOX = 4
FONT_SCALE = 1.0
THICK = 2
ALPHA = 0.6

def draw_label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, THICK)
    bg_x1, bg_y1 = x, max(0, y - th - 6)
    bg_x2, bg_y2 = x + tw + 8, y + 2
    overlay = img.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0,0,0), -1)
    cv2.addWeighted(overlay, ALPHA, img, 1-ALPHA, 0, img)
    cv2.putText(img, text, (x+4, y-4), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0,0,0), THICK+2, cv2.LINE_AA)
    cv2.putText(img, text, (x+4, y-4), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, THICK, cv2.LINE_AA)

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Step2: YOLOv8(conf=0.70) 직접 탐지 → ROI 값 산출 → Cohen's d 최대 (method, metric) 선택")
    ap.add_argument("--weights", type=str, required=True, help="YOLO 가중치 경로")
    ap.add_argument("--roots", nargs="+", required=True, help="분석할 이미지 폴더/파일들 (pair/solo 혼합 가능)")
    ap.add_argument("--out_dir", type=str, required=True, help="결과 저장 폴더")
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--class_tube_name", type=str, default="tube")
    ap.add_argument("--class_roi_name", type=str, default="roi")
    ap.add_argument("--max_tubes", type=int, default=2, help="상위 N개 튜브만 고려 (기본 2)")
    ap.add_argument("--methods", nargs="+", default=["G","HSV_V","GRAY","HSV_S","G_norm","G_ratio","ExG"])
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    viz_dir = ensure_dir(out_dir / "viz")
    values_csv = out_dir / "channel_scan_values.csv"
    rep_csv    = out_dir / "channel_scan_report.csv"
    best_json  = out_dir / "best_channel.json"

    model = YOLO(str(args.weights))
    img_paths = [p for p in list_images(args.roots) if not is_in_test_all(str(p))]

    rows = []  # per-ROI rows for stats

    for i, img_path in enumerate(img_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] cannot read: {img_path}")
            continue

        # ---- YOLO predict (conf 고정 0.70) ----
        r = model.predict(
            source=str(img_path),
            conf=0.70, iou=args.iou, imgsz=args.imgsz,
            device=args.device, verbose=False
        )[0]

        names = r.names
        inv = {v: k for k, v in names.items()}
        if args.class_tube_name not in inv or args.class_roi_name not in inv:
            print(f"[WARN] class name not found in model: {names}")
            continue
        tube_id = inv[args.class_tube_name]
        roi_id  = inv[args.class_roi_name]

        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0,4))
        clses = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros((0,), dtype=int)
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,), dtype=float)

        tubes_all, tubes_conf = [], []
        rois_all,  rois_conf  = [], []
        for b, c, cf in zip(boxes, clses, confs):
            if c == tube_id:
                tubes_all.append(to_xyxy(b)); tubes_conf.append(float(cf))
            elif c == roi_id:
                rois_all.append(to_xyxy(b));  rois_conf.append(float(cf))

        # 상위 N개 튜브 선택
        idx_sorted = sorted(range(len(tubes_all)), key=lambda k: tubes_conf[k], reverse=True)
        sel_idx = idx_sorted[: min(args.max_tubes, len(idx_sorted))]
        sel_tubes = [tubes_all[k] for k in sel_idx]
        sel_tubes_conf = [tubes_conf[k] for k in sel_idx]

        # 각 선택 튜브마다 '포함되는 ROI들 중 최고 conf' 하나 pick
        tube_roi_pairs = []  # [(tube_xyxy, tube_conf, best_roi_xyxy, best_roi_conf)]
        for k in sel_idx:
            tb = tubes_all[k]
            contained = [(ri, rc) for ri, rc in zip(rois_all, rois_conf) if inside(ri, tb)]
            if contained:
                contained.sort(key=lambda x: x[1], reverse=True)
                best_ri, best_rc = contained[0]
            else:
                best_ri, best_rc = None, None
            tube_roi_pairs.append((tb, tubes_conf[k], best_ri, best_rc))

        # 역할 부여 & 값 계산
        img_role = None
        if is_pair_image(str(img_path)):
            img_role = "pair"
            # 수직 위치로 정렬: top=neg, bottom=pos
            tri = []
            for (tb, tcf, rb, rcf) in tube_roi_pairs:
                if rb is None: continue
                tri.append((center_y(rb), tb, tcf, rb, rcf))
            tri.sort(key=lambda x: x[0])  # y-center asc
            # 위 neg, 아래 pos
            assigned = []
            if len(tri) >= 1: assigned.append(("neg", tri[0][1], tri[0][2], tri[0][3], tri[0][4]))
            if len(tri) >= 2: assigned.append(("pos", tri[1][1], tri[1][2], tri[1][3], tri[1][4]))
        elif is_solo_image(str(img_path)):
            img_role = "solo"
            label = solo_label_from_path(str(img_path))
            if label not in {"pos","neg"}:
                assigned = []  # 모호 → 스킵
            else:
                assigned = []
                for (tb, tcf, rb, rcf) in tube_roi_pairs:
                    if rb is None: continue
                    assigned.append((label, tb, tcf, rb, rcf))
        else:
            assigned = []

        # per-ROI rows 생성 (여러 methods)
        for (label, tb, tcf, rb, rcf) in assigned:
            crop = safe_crop(img, rb)
            if crop is None: continue
            for m in args.methods:
                val = compute_value(m, crop)
                if val is None: continue
                rows.append({
                    "image_id": Path(img_path).stem,
                    "image_path": str(img_path),
                    "role": img_role,
                    "label": label,            # pos/neg
                    "method": m,
                    "tube_conf": f"{tcf:.6f}",
                    "roi_conf":  f"{(rcf if rcf is not None else 0.0):.6f}",
                    "mean_au":   f"{val['mean_au']:.6f}",
                    "p95_au":    f"{val['p95_au']:.6f}",
                })

        # ---- 시각화 저장 ----
        canvas = img.copy()
        # tubes
        for (tb, tcf, rb, rcf) in tube_roi_pairs:
            x1,y1,x2,y2 = tb
            cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,255,0), BOX)
            draw_label(canvas, f"T{tcf:.2f}", x1, y1, (0,255,0))
        # rois
        for (tb, tcf, rb, rcf) in tube_roi_pairs:
            if rb is None: continue
            x1,y1,x2,y2 = rb
            cv2.rectangle(canvas, (x1,y1), (x2,y2), (0,0,255), BOX)
            draw_label(canvas, f"R{(rcf if rcf is not None else 0.0):.2f}", x1, y1, (0,0,255))
        out_img = viz_dir / f"{Path(img_path).stem}_viz.jpg"
        cv2.imwrite(str(out_img), canvas)

        if i % 20 == 0 or i == len(img_paths):
            print(f"[{i}/{len(img_paths)}] processed {img_path} -> {out_img}")

    # ---- 저장: per-ROI values ----
    with open(values_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "image_id","image_path","role","label","method",
            "tube_conf","roi_conf","mean_au","p95_au"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- Cohen's d 계산 ----
    buckets = {}
    for r in rows:
        m = r["method"]
        buckets.setdefault(m, {"pos_mean":[], "neg_mean":[], "pos_p95":[], "neg_p95":[]})
        mv = float(r["mean_au"])
        pv = float(r["p95_au"])
        if r["label"] == "pos":
            buckets[m]["pos_mean"].append(mv)
            buckets[m]["pos_p95"].append(pv)
        elif r["label"] == "neg":
            buckets[m]["neg_mean"].append(mv)
            buckets[m]["neg_p95"].append(pv)

    report = []
    best = {"method": None, "metric": None, "d": -1e9}
    for m, B in buckets.items():
        d_mean = cohens_d(B["pos_mean"], B["neg_mean"])
        d_p95  = cohens_d(B["pos_p95"],  B["neg_p95"])
        report.append({"method": m, "metric": "mean", "cohens_d": d_mean})
        report.append({"method": m, "metric": "p95",  "cohens_d": d_p95})
        for metric, dval in [("mean", d_mean), ("p95", d_p95)]:
            if dval is not None and not np.isnan(dval) and dval > best["d"]:
                best = {"method": m, "metric": metric, "d": float(dval)}

    with open(rep_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method","metric","cohens_d"])
        w.writeheader()
        for r in report:
            cd = r["cohens_d"]
            w.writerow({"method": r["method"], "metric": r["metric"], "cohens_d": f"{cd:.6f}" if cd==cd else ""})

    with open(best_json, "w", encoding="utf-8") as f:
        json.dump({"best": best}, f, indent=2, ensure_ascii=False)

    print("\n[Step2] Channel/Method Scan with YOLOv8 (conf=0.70 fixed, test_all excluded)")
    print(f"- values : {values_csv}")
    print(f"- report : {rep_csv}")
    print(f"- best   : {best_json}  -> {best}")

if __name__ == "__main__":
    main()
