# app.py
# ------------------------------------------------------------
# 스마트폰 기반 임질 진단 시스템 (YOLOv8 + G(p95) + Il/Iu ratio)
# - Il/Iu, 판정 기준, ROI 측정 방식 유지
# - 오류/주의(행동지시형) 강화
# - Gemini 대화: 검사결과를 기억하고 답변
# - 병원 검색: Kakao Local API 사용(기본), 의학 최신정보는 Google CSE(선택)
# - 하단에 powered by Gemini <model>
# ------------------------------------------------------------

import hashlib
import os
import json
import numpy as np
import cv2
import requests
import streamlit as st
from importlib.metadata import version as pkg_version

# ---------------- YOLO ----------------
try:
    from ultralytics import YOLO
except Exception:
    st.error("ultralytics가 필요합니다. `pip install ultralytics` 후 다시 실행하세요.")
    raise

# ------------------- 전역 고정 파라미터 (변경 금지) -------------------
MODEL_PATH_DEFAULT = "models/new_weights.pt"
CONF_MIN = 0.70
IOU = 0.50
IMG_SIZE = 640

# 임계 설정 (사용자 고정값)
RATIO_THR = 1.148       # Il/Iu 임계
ABS_NEG_CUTOFF = 221.0  # 상단(음성튜브) 절대 밝기 컷오프

# 렌더링 옵션
BOX_THICK = 4
FONT_SCALE = 1.15
FONT_THICK = 3
LABEL_ALPHA = 0.65

# 색상 (BGR)
COLOR_TUBE = (0, 255, 0)
COLOR_ROI  = (255, 0, 255)
COLOR_TEXT = (255, 255, 255)

# ------------------- 세션 기본값 -------------------
if "gemini_chat" not in st.session_state:
    st.session_state["gemini_chat"] = None
if "last_img_hash" not in st.session_state:
    st.session_state["last_img_hash"] = None
if "gemini_report" not in st.session_state:
    st.session_state["gemini_report"] = None
if "chat_ui" not in st.session_state:
    st.session_state["chat_ui"] = []
if "gemini_model" not in st.session_state:
    st.session_state["gemini_model"] = "gemini-2.5-flash"

# ---------------- Google Custom Search (선택) ----------------
def cse_available() -> bool:
    return bool(st.secrets.get("GOOGLE_API_KEY")) and bool(st.secrets.get("GOOGLE_CSE_ID"))

def google_cse_search(query: str, num: int = 6) -> list:
    api_key = st.secrets.get("GOOGLE_API_KEY")
    cse_id  = st.secrets.get("GOOGLE_CSE_ID")
    if not (api_key and cse_id):
        return []
    try:
        r = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": api_key, "cx": cse_id, "q": query, "num": num, "hl": "ko"},
            timeout=6,
        )
        if not r.ok:
            return []
        data = r.json()
        results = []
        for it in data.get("items", []):
            results.append({
                "title": it.get("title"),
                "snippet": it.get("snippet"),
                "link": it.get("link"),
            })
        return results
    except Exception:
        return []

# ---------------- Kakao Local API (병원 검색) ----------------
def kakao_search_places(query: str, size: int = 5) -> list:
    """
    카카오 키워드 검색 결과 반환.
    반환: [{name, address, phone, url}] 리스트
    """
    kakao_key = st.secrets.get("KAKAO_API_KEY")
    if not kakao_key:
        return []

    headers = {"Authorization": f"KakaoAK {kakao_key}"}
    try:
        r = requests.get(
            "https://dapi.kakao.com/v2/local/search/keyword.json",
            headers=headers,
            params={"query": query, "size": size},
            timeout=6,
        )
        if not r.ok:
            return []
        docs = r.json().get("documents", [])
        out = []
        for d in docs:
            name = d.get("place_name", "")
            addr = d.get("road_address_name") or d.get("address_name") or ""
            phone = d.get("phone") or ""
            pid = d.get("id")
            url = f"http://place.map.kakao.com/{pid}" if pid else (d.get("place_url") or "")
            out.append({"name": name, "address": addr, "phone": phone, "url": url})
        return out
    except Exception:
        return []

# ---------------- 유틸 ----------------
def to_xyxy(b):
    return [int(float(b[0])), int(float(b[1])), int(float(b[2])), int(float(b[3]))]

def center_y(b):
    return (b[1] + b[3]) / 2.0

def inside(inner, outer):
    x1, y1, x2, y2 = inner
    X1, Y1, X2, Y2 = outer
    return (x1 >= X1 and y1 >= Y1 and x2 <= X2 and y2 <= Y2)

def safe_crop(img, xyxy):
    if img is None or xyxy is None:
        return None
    x1,y1,x2,y2 = [int(v) for v in xyxy]
    H,W = img.shape[:2]
    x1=max(0,x1); y1=max(0,y1); x2=min(W-1,x2); y2=min(H-1,y2)
    if x2<=x1 or y2<=y1: return None
    return img[y1:y2, x1:x2]

def g_p95(crop_bgr):
    if crop_bgr is None:
        return np.nan
    G = crop_bgr[:,:,1].astype(np.float32)
    return float(np.percentile(G, 95.0))

def draw_label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICK)
    bg_x1, bg_y1 = x, max(0, y - th - 8)
    bg_x2, bg_y2 = x + tw + 12, y + 4
    overlay = img.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, LABEL_ALPHA, img, 1 - LABEL_ALPHA, 0, img)
    cv2.putText(img, text, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0,0,0), FONT_THICK+2, cv2.LINE_AA)
    cv2.putText(img, text, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR_TEXT, FONT_THICK, cv2.LINE_AA)

def draw_box(img, xyxy, color, label=None, show=True):
    x1,y1,x2,y2 = [int(v) for v in xyxy]
    if show:
        cv2.rectangle(img, (x1,y1), (x2,y2), color, BOX_THICK)
    if label:
        draw_label(img, label, x1, y1, color)

def show_bgr_image_safe(img_bgr, caption=None):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption=caption, width=400)

# ---------------- 탐지 (YOLOv8 + G(p95)) ----------------
def detect_pair_and_measure(img_bgr, model):
    r = model.predict(source=img_bgr, imgsz=IMG_SIZE, conf=CONF_MIN, iou=IOU, verbose=False)[0]
    names = r.names
    inv = {v:k for k,v in names.items()}
    if "tube" not in inv or "roi" not in inv:
        raise RuntimeError(f"모델 클래스에 'tube' 또는 'roi'가 없습니다. names={names}")

    tube_id = inv["tube"]; roi_id = inv["roi"]
    boxes = r.boxes.xyxy.cpu().numpy()
    clses = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()

    tubes, tubes_conf = [], []
    rois,  rois_conf  = [], []
    for b, c, cf in zip(boxes, clses, confs):
        if c == tube_id:
            tubes.append(to_xyxy(b)); tubes_conf.append(float(cf))
        elif c == roi_id:
            rois.append(to_xyxy(b));  rois_conf.append(float(cf))

    pairs = []
    for ti, tb in enumerate(tubes):
        contained = [(ri, rc) for ri, rc in zip(rois, rois_conf) if inside(ri, tb)]
        if contained:
            contained.sort(key=lambda x: x[1], reverse=True)
            best_ri, best_rc = contained[0]
        else:
            best_ri, best_rc = None, None
        pairs.append((tb, tubes_conf[ti], best_ri, best_rc))

    tri = []
    for (tb, tcf, rb, rcf) in pairs:
        if rb is not None:
            cy = center_y(rb)
            tri.append((cy, tb, tcf, rb, rcf))
    tri.sort(key=lambda x: x[0])

    upper, lower = (tri[0] if len(tri) >= 1 else None), (tri[1] if len(tri) >= 2 else None)

    Iu = Il = np.nan
    if upper:  Iu = g_p95(safe_crop(img_bgr, upper[3]))
    if lower:  Il = g_p95(safe_crop(img_bgr, lower[3]))
    ratio = (Il / Iu) if (np.isfinite(Iu) and Iu > 0) else np.nan

    # ----- 오류/주의 가이드 (행동 지시형) -----
    notes = []
    if len(tubes) == 0 or all(cf < CONF_MIN for cf in tubes_conf):
        notes.append(
            "튜브가 잘 잡히지 않습니다: 초점이 맞지 않았거나 강한 빛반사가 있을 수 있어요. "
            "카메라를 10–15cm 거리에서 정면에 가깝게 두고 렌즈를 닦은 뒤, 상부 조명이 비껴가도록 각도를 약간 바꿔 재촬영해 주세요."
        )
    if (upper is None or lower is None):
        notes.append(
            "표적 영역이 한쪽만 잡히거나 빠졌습니다: 용액이 흩어진(splash) 상황일 수 있어요. "
            "튜브를 수직으로 세우고 바닥을 2–3회 가볍게 톡톡 쳐서 용액이 바닥으로 모이게 한 뒤, 거품이 가라앉으면 재촬영해 주세요."
        )
    if np.isfinite(Iu) and Iu >= ABS_NEG_CUTOFF:
        notes.append(
            "상단(기준) 밝기가 비정상적으로 높습니다. 상단에는 반드시 음성 대조(NC)를 사용하고, 반사광이 강하면 각도를 조정해 재촬영해 주세요."
        )
    if not np.isfinite(ratio):
        notes.append(
            "Il/Iu 비율을 계산할 수 없습니다: 위 안내대로 재촬영 후 다시 시도해 주세요."
        )

    is_positive = (np.isfinite(ratio) and ratio >= RATIO_THR)

    viz_items = dict(
        tubes=[(tb, tcf) for (tb, tcf, _, _) in pairs],
        rois=[(rb, rcf) for (_, _, rb, rcf) in pairs if rb is not None],
        upper=upper, lower=lower
    )
    return Iu, Il, ratio, is_positive, notes, viz_items

def overlay_visual(img_bgr, viz_items):
    canvas = img_bgr.copy()
    for tb, tcf in viz_items["tubes"]:
        show = (tcf >= CONF_MIN)
        draw_box(canvas, tb, COLOR_TUBE, label=f"CONF {tcf:.2f}", show=show)
    for rb, rcf in viz_items["rois"]:
        show = (rcf >= CONF_MIN)
        draw_box(canvas, rb, COLOR_ROI, label=f"CONF {rcf:.2f}", show=show)
    return canvas

# ---------------- Gemini (항상-초기화 + 안전 폴백) ----------------
def _get_gemini_model():
    try:
        import google.generativeai as genai
    except Exception:
        st.warning("google-generativeai 패키지가 필요합니다. `pip install google-generativeai`")
        return None, None
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        return None, None
    genai.configure(api_key=api_key)
    model_name = st.session_state.get("gemini_model", "gemini-2.5-flash")
    return genai.GenerativeModel(model_name), model_name

def _ensure_gemini_chat(context_ko: str = "현재 이미지 컨텍스트 없음."):
    if st.session_state.get("gemini_chat") is not None:
        return st.session_state["gemini_chat"]
    model, _ = _get_gemini_model()
    if model is None:
        return None
    system_prompt = (
        "역할: 임질(Neisseria gonorrhoeae) 체외진단 앱의 한국어 어시스턴트.\n"
        "원칙: 짧고 정확, 일반인 친화 설명. 확진/처방 지시는 금지.\n"
        f"[현재 측정 요약]\n{context_ko}\n"
    )
    try:
        chat = model.start_chat(history=[
            {"role": "user", "parts": system_prompt},
            {"role": "model", "parts": "컨텍스트를 기억했습니다. 질문을 주세요."}
        ])
        st.session_state["gemini_chat"] = chat
        return chat
    except Exception:
        return None

def gemini_safe_reply(prompt: str, context_ko: str = "현재 이미지 컨텍스트 없음.") -> str:
    model, _ = _get_gemini_model()
    if model is None:
        return "(Gemini 비활성화)"

    chat = _ensure_gemini_chat(context_ko)
    if chat is not None:
        try:
            resp = chat.send_message(prompt)
            return getattr(resp, "text", "") or "(빈 응답)"
        except Exception:
            st.session_state["gemini_chat"] = None
            chat = _ensure_gemini_chat(context_ko)
            if chat is not None:
                try:
                    resp = chat.send_message(prompt)
                    return getattr(resp, "text", "") or "(빈 응답)"
                except Exception:
                    pass
    try:
        resp = model.generate_content(prompt)
        return getattr(resp, "text", "") or "(빈 응답)"
    except Exception as e:
        return f"(Gemini 응답 실패: {e})"

# ----- (구) 질의 전처리: 지명/과목 뽑기 함수 (필요시 재사용 가능) -----
def gemini_normalize_location_query(user_msg: str) -> dict:
    """
    LLM에게 '분당 근처 산부인과' 같은 문장에서 지명/과목 추출을 맡긴다.
    반환 예: {"place": "분당", "specialty": "산부인과"}
    (현재는 intent 분류용 함수에서 직접 처리하지만, 혹시 몰라 남겨둠)
    """
    model, _ = _get_gemini_model()
    if model is None:
        # LLM이 없으면 단순 휴리스틱
        return {"place": user_msg.replace("근처", "").replace("주변", "").replace("가까운", "").replace("추천", "").strip(),
                "specialty": ""}

    sys = (
        "너는 사용자의 병원 찾기 문장에서 '지명'과 '진료과목'만 뽑아 JSON으로만 답한다. "
        "불용어(근처, 주변, 가까운, 추천, 알려줘 등)는 무시한다. "
        "예: '분당 근처 산부인과 추천해줘' → {\"place\":\"분당\",\"specialty\":\"산부인과\"}"
    )
    try:
        resp = model.generate_content(f"{sys}\n문장: {user_msg}")
        txt = getattr(resp, "text", "") or ""
        s = txt.strip()
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            j = json.loads(s[start:end+1])
            return {"place": j.get("place", ""), "specialty": j.get("specialty", "")}
    except Exception:
        pass
    return {"place": user_msg, "specialty": ""}

# ----- 새 질의 전처리: intent + 지명/과목 분류 -----
def classify_query_with_gemini(user_msg: str) -> dict:
    """
    intent: 'general' | 'hospital_search' | 'med_
