# step2_2.py
# 목적:
#  - step2_fluorescence_analysis.py와 동일한 방식으로
#    YOLOv8(conf=0.70 고정)으로 tube/ROI 검출
#  - solo/pair 이미지에서 ROI를 매칭하고, G 채널 기준 mean_au, p95_au 계산
#  - label(neg/pos) 별 G_p95 분포를 모아서 Welch t-test(p-value) 계산
#  - 결과를 out_dir 아래 CSV/JSON 및 viz 이미지로 저장
#
# 실행 예:
# py "C:\n.gonorrhea_diagnostic_app\analysis_code\step2_2.py" ^
#   --weights "C:\n.gonorrhea_diagnostic_app\models\new_weights.pt" ^
#   --roots  "C:\n.gonorrhea_diagnostic_app\dataset\solo\train\neg" ^
#            "C:\n.gonorrhea_diagnostic_app\dataset\solo\train\pos" ^
#            "C:\n.gonorrhea_diagnostic_app\dataset\solo\test\neg"  ^
#            "C:\n.gonorrhea_diagnostic_app\dataset\solo\test\pos"  ^
#   --out_dir "C:\n.gonorrhea_diagnostic_app\analysis_output\step2_2" ^
#   --imgsz 640 --iou 0.50

import argparse
from pathlib import Path
import csv
import json
import cv2
import numpy as np

# ----------------- YOLO -----------------
try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError("ultralytics가 필요합니다. pip install ultralytics") from e

# Welch t-test를 위해 scipy 사용
try:
    from scipy import stats
except Exception as e:
    raise RuntimeError("scipy가 필요합니다. pip install scipy") from e

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
    # 기존 코드와 동일: 경로 중 'test_all' 포함되면 제외
    return any("test_all" in part for part in path_parts_lower(path))

def is_pair_image(path: str) -> bool:
    if is_in_test_all(path): 
        return False
    parts = path_parts_lower(path)
    return ("pair" in parts) or any("neg_pos" in s for s in parts)

def is_solo_image(path: str) -> bool:
    if is_in_test_all(path): 
        return False
    parts = path_parts_lower(path)
    return "solo" in parts

def solo_label_from_path(path: str):
    # solo 이미지에서 폴더명으로 pos/neg 라벨 결정 (기존과 동일)
    if not is_solo_image(path): 
        return None
    parts = path_parts_lower(path)
    has_pos = any(part == "pos" for part in parts)
    has_neg = any(part == "neg" for part in parts)
    if has_pos and not has_neg: 
        return "pos"
    if has_neg and not has_pos: 
        return "neg"
    return None  # 모호하면 제외

# ----------------- geometry -----------------
def to_xyxy(b):
    return [int(float(b[0])), int(float(b[1])), int(float(b[2])), int(float(b[3]))]

def inside(inner, outer):
    x1, y1, x2, y2 = inner
    X1, Y1, X2, Y2 = outer
    return (x1 >= X1 and y1 >= Y1 and x2 <= X2 and y2 <= Y2)

def center_y(b):
    return (b[1] + b[3]) / 2.0

def safe_crop(img, xyxy):
    if img is None or xyxy is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    H, W = img.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(W - 1, x2); y2 = min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

# ----------------- intensity methods -----------------
def get_channels(img_bgr):
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    B, G, R = cv2.split(img_bgr)
    H, S, V = cv2.split(hsv)
    return {"B": B, "G": G, "R": R, "HSV_S": S, "HSV_V": V, "GRAY": gray}

def compute_value(method: str, crop_bgr):
    """
    step2_fluorescence_analysis.py와 동일한 방식으로
    method="G"일 때 mean_au, p95_au 계산.
    (다른 method 분기도 그대로 두지만, 여기서는 G만 사용)
    """
    if crop_bgr is None:
        return None
    chs = get_channels(crop_bgr)
    B, G, R = chs["B"], chs["G"], chs["R"]
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
        M = G.astype(np.float32) / (R.astype(np.float32) + G.astype(np.float32) + B.astype(np.float32) + eps)
        M *= 255.0
    elif method == "G_ratio":    # G / (R+B) -> normalize to [0..255]
        M = G.astype(np.float32) / (R.astype(np.float32) + B.astype(np.float32) + eps)
        M = np.clip(M, 0, np.percentile(M, 99.9))
        M = (M / (np.max(M) + eps)) * 255.0
    elif method == "ExG":        # 2G - R - B -> [0..255] by contrast stretch
        M = 2.0 * G.astype(np.float32) - R.astype(np.float32) - B.astype(np.float32)
        lo, hi = np.percentile(M, 1.0), np.percentile(M, 99.0)
        if hi <= lo:
            hi = lo + 1.0
        M = np.clip((M - lo) / (hi - lo), 0.0, 1.0) * 255.0
    else:
        return None

    mean_v = float(np.mean(M))
    p95_v  = float(np.percentile(M, 95.0))
    return {"mean_au": mean_v, "p95_au": p95_v}

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
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, ALPHA, img, 1 - ALPHA, 0, img)
    cv2.putText(img, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, (0, 0, 0), THICK + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE, color, THICK, cv2.LINE_AA)

# ----------------- Welch t-test -----------------
def welch_ttest(pos_vals, neg_vals):
    pos_vals = np.asarray(pos_vals, dtype=float)
    neg_vals = np.asarray(neg_vals, dtype=float)

    if len(pos_vals) < 2 or len(neg_vals) < 2:
        raise ValueError("Welch t-test를 위해서는 각 그룹에 최소 2개 이상의 샘플이 필요합니다.")

    # equal_var=False 로 Welch t-test 수행
    t_stat, p_value = stats.ttest_ind(pos_vals, neg_vals, equal_var=False)

    pos_mean = float(np.mean(pos_vals))
    neg_mean = float(np.mean(neg_vals))
    pos_std  = float(np.std(pos_vals, ddof=1))
    neg_std  = float(np.std(neg_vals, ddof=1))

    return {
        "metric": "G_p95_au",
        "pos_n": int(len(pos_vals)),
        "neg_n": int(len(neg_vals)),
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "pos_std": pos_std,
        "neg_std": neg_std,
        "t_stat": float(t_stat),
        "p_value_two_sided": float(p_value),
        "test": "Welch t-test (scipy.stats.ttest_ind, equal_var=False)"
    }

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(
        description="YOLOv8(conf=0.70) → ROI 검출 → G_p95 intensity 계산 → Welch t-test(p-value) 산출"
    )
    ap.add_argument("--weights", type=str, required=True, help="YOLO 가중치 경로")
    ap.add_argument("--roots", nargs="+", required=True,
                    help="분석할 이미지 폴더/파일들 (pair/solo 혼합 가능, test_all은 자동 제외)")
    ap.add_argument("--out_dir", type=str, required=True, help="결과 저장 폴더")
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--class_tube_name", type=str, default="tube")
    ap.add_argument("--class_roi_name", type=str, default="roi")
    ap.add_argument("--max_tubes", type=int, default=2,
                    help="상위 N개 튜브만 고려 (기본 2)")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    viz_dir = ensure_dir(out_dir / "viz")
    values_csv = out_dir / "Gp95_values.csv"
    stats_csv  = out_dir / "Gp95_stats.csv"
    stats_json = out_dir / "Gp95_stats.json"

    # YOLO 모델 로드
    model = YOLO(str(args.weights))

    # 이미지 리스트 (test_all은 자동 제외)
    img_paths = [p for p in list_images(args.roots) if not is_in_test_all(str(p))]

    rows = []  # per-ROI row 저장 (G_mean, G_p95)

    for i, img_path in enumerate(img_paths, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[WARN] cannot read: {img_path}")
            continue

        # ---- YOLO predict (conf 고정 0.70) ----
        r = model.predict(
            source=str(img_path),
            conf=0.70,          # 고정
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False
        )[0]

        names = r.names
        inv = {v: k for k, v in names.items()}
        if args.class_tube_name not in inv or args.class_roi_name not in inv:
            print(f"[WARN] class name not found in model: {names}")
            continue
        tube_id = inv[args.class_tube_name]
        roi_id  = inv[args.class_roi_name]

        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        clses = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros((0,), dtype=int)
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,), dtype=float)

        tubes_all, tubes_conf = [], []
        rois_all,  rois_conf  = [], []
        for b, c, cf in zip(boxes, clses, confs):
            if c == tube_id:
                tubes_all.append(to_xyxy(b)); tubes_conf.append(float(cf))
            elif c == roi_id:
                rois_all.append(to_xyxy(b));  rois_conf.append(float(cf))

        # 상위 N개 튜브 선택 (conf 기준)
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

        # 이미지의 role/label 정하기 (pair/solo)
        img_role = None
        if is_pair_image(str(img_path)):
            img_role = "pair"
            # 수직 위치로 정렬: top=neg, bottom=pos
            tri = []
            for (tb, tcf, rb, rcf) in tube_roi_pairs:
                if rb is None:
                    continue
                tri.append((center_y(rb), tb, tcf, rb, rcf))
            tri.sort(key=lambda x: x[0])  # y-center asc
            assigned = []
            if len(tri) >= 1:
                assigned.append(("neg", tri[0][1], tri[0][2], tri[0][3], tri[0][4]))
            if len(tri) >= 2:
                assigned.append(("pos", tri[1][1], tri[1][2], tri[1][3], tri[1][4]))
        elif is_solo_image(str(img_path)):
            img_role = "solo"
            label = solo_label_from_path(str(img_path))
            if label not in {"pos", "neg"}:
                assigned = []  # 모호 → 스킵
            else:
                assigned = []
                for (tb, tcf, rb, rcf) in tube_roi_pairs:
                    if rb is None:
                        continue
                    assigned.append((label, tb, tcf, rb, rcf))
        else:
            assigned = []

        # per-ROI rows 생성 (여기서는 method="G"만 사용)
        for (label, tb, tcf, rb, rcf) in assigned:
            crop = safe_crop(img, rb)
            if crop is None:
                continue
            val = compute_value("G", crop)
            if val is None:
                continue
            rows.append({
                "image_id": Path(img_path).stem,
                "image_path": str(img_path),
                "role": img_role,
                "label": label,            # pos/neg
                "tube_conf": f"{tcf:.6f}",
                "roi_conf":  f"{(rcf if rcf is not None else 0.0):.6f}",
                "G_mean_au": f"{val['mean_au']:.6f}",
                "G_p95_au":  f"{val['p95_au']:.6f}",
            })

        # ---- 시각화 저장 ----
        canvas = img.copy()
        # tubes
        for (tb, tcf, rb, rcf) in tube_roi_pairs:
            x1, y1, x2, y2 = tb
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), BOX)
            draw_label(canvas, f"T{tcf:.2f}", x1, y1, (0, 255, 0))
        # rois
        for (tb, tcf, rb, rcf) in tube_roi_pairs:
            if rb is None:
                continue
            x1, y1, x2, y2 = rb
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), BOX)
            draw_label(canvas, f"R{(rcf if rcf is not None else 0.0):.2f}", x1, y1, (0, 0, 255))
        out_img = viz_dir / f"{Path(img_path).stem}_viz.jpg"
        cv2.imwrite(str(out_img), canvas)

        if i % 20 == 0 or i == len(img_paths):
            print(f"[{i}/{len(img_paths)}] processed {img_path} -> {out_img}")

    # ---- per-ROI values CSV 저장 ----
    with open(values_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "image_id", "image_path", "role", "label",
            "tube_conf", "roi_conf",
            "G_mean_au", "G_p95_au"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- Welch t-test용 데이터 추출 (G_p95 기준) ----
    pos_vals = []
    neg_vals = []
    for r in rows:
        try:
            v = float(r["G_p95_au"])
        except Exception:
            continue
        if r["label"] == "pos":
            pos_vals.append(v)
        elif r["label"] == "neg":
            neg_vals.append(v)

    if len(pos_vals) < 2 or len(neg_vals) < 2:
        print("[WARN] Welch t-test를 수행하기에 pos/neg 샘플 수가 부족합니다.")
        print(f" pos_n={len(pos_vals)}, neg_n={len(neg_vals)}")
        return

    stats_res = welch_ttest(pos_vals, neg_vals)

    # ---- stats CSV/JSON 저장 ----
    # CSV
    with open(stats_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "metric", "pos_n", "neg_n",
            "pos_mean", "neg_mean",
            "pos_std", "neg_std",
            "t_stat", "p_value_two_sided", "test"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(stats_res)

    # JSON
    with open(stats_json, "w", encoding="utf-8") as f:
        json.dump(stats_res, f, indent=2, ensure_ascii=False)

    # ---- 콘솔 출력 ----
    print("\n[Step2_2] G_p95 기반 Welch t-test 결과")
    print(f" - pos_n : {stats_res['pos_n']}")
    print(f" - neg_n : {stats_res['neg_n']}")
    print(f" - pos_mean (G_p95_au): {stats_res['pos_mean']:.6f}")
    print(f" - neg_mean (G_p95_au): {stats_res['neg_mean']:.6f}")
    print(f" - t_stat            : {stats_res['t_stat']:.6f}")
    print(f" - p_value(two-sided): {stats_res['p_value_two_sided']:.6e}")
    print("\n[Outputs]")
    print(f" - per-ROI values : {values_csv}")
    print(f" - Welch t-test   : {stats_csv}")
    print(f" - Welch t-test   : {stats_json}")
    print(f" - viz images     : {viz_dir}")

if __name__ == "__main__":
    main()
