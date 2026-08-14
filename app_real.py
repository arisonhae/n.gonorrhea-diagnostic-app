# app.py (또는 app_real.py)
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
RRATIO_THR = 1.1162     # step4: 221.0 / median(pair NC, n=44). 원본 1.148은 n=20 기준 — CHANGELOG 참고
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
    # 카카오 키워드 검색 결과 반환.
    # 반환: [{name, address, phone, url}] 리스트
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
    # LLM에게 '분당 근처 산부인과' 같은 문장에서 지명/과목 추출을 맡긴다.
    # 반환 예: {"place": "분당", "specialty": "산부인과"}
    model, _ = _get_gemini_model()
    if model is None:
        # LLM이 없으면 단순 휴리스틱
        return {
            "place": user_msg.replace("근처", "").replace("주변", "").replace("가까운", "").replace("추천", "").strip(),
            "specialty": ""
        }

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
    # intent: 'general' | 'hospital_search' | 'med_news'
    # place: 지명 (없으면 "")
    # specialty: 진료과 (없으면 "")
    model, _ = _get_gemini_model()
    if model is None:
        # Gemini 사용 불가 시 기본은 일반 질문으로 처리
        return {"intent": "general", "place": "", "specialty": ""}

    sys = (
        "너는 사용자의 문장을 '의도(intent)'와 '지명(place)/진료과목(specialty)'로 분류하는 도우미야.\n"
        "반드시 JSON 한 줄만 출력해.\n"
        "intent는 다음 중 하나여야 한다:\n"
        "  - 'general': 일반적인 질문(증상, 무증상, 병원 가야 하는지, 예방, 경과 등)\n"
        "  - 'hospital_search': 실제로 특정 지역의 병원/의원/산부인과/비뇨의학과를 찾아달라는 경우\n"
        "  - 'med_news': 최신 치료 가이드라인, 논문, 뉴스 등 의학 최신 정보를 묻는 경우\n\n"
        "각 필드는 항상 존재해야 한다. 예를 들어:\n"
        "예시1: '분당 산부인과 추천해줘' -> "
        "{\"intent\":\"hospital_search\",\"place\":\"분당\",\"specialty\":\"산부인과\"}\n"
        "예시2: '난 아무 증상이 없는데, 병원을 가야 해?' -> "
        "{\"intent\":\"general\",\"place\":\"\",\"specialty\":\"\"}\n"
        "예시3: '임질 최신 치료 가이드라인 알려줘' -> "
        "{\"intent\":\"med_news\",\"place\":\"\",\"specialty\":\"\"}\n"
    )

    try:
        resp = model.generate_content(f"{sys}\n\n사용자 문장: {user_msg}")
        txt = getattr(resp, "text", "") or ""
        s = txt.strip()
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            j = json.loads(s[start:end+1])
            return {
                "intent": j.get("intent", "general"),
                "place": j.get("place", ""),
                "specialty": j.get("specialty", "")
            }
    except Exception:
        pass
    return {"intent": "general", "place": "", "specialty": ""}

# ---------------- 보고서 / 대화 프롬프트 ----------------
def make_report_prompt(Iu, Il, ratio, thr, is_pos, notes):
    ratio_txt = f"{ratio:.3f}" if np.isfinite(ratio) else "계산불가"
    judge = '양성' if is_pos else ('음성' if np.isfinite(ratio) else '판정불가')
    return (
        "다음 데이터를 바탕으로 환자용 한국어 요약 보고서를 작성하세요.\n"
        "스타일: 제목 1줄 + 간단 근거 + 방법 설명 + 오류/주의(해결 포함) + 다음 단계 + 면책.\n"
        f"- 상단 밝기 Iu={Iu:.2f}, 하단 밝기 Il={Il:.2f}, 비율 Il/Iu={ratio_txt}, 임계={thr:.3f}\n"
        f"- 판정: {judge}\n"
        f"- 참고 노트: {notes}\n\n"
        "구성:\n"
        "1) 한줄 요약: 양성/음성과 근거(Il/Iu vs 임계)\n"
        "2) 결과 해석(일반어): Iu/Il이 무엇인지와 이번 숫자의 의미\n"
        "3) **측정 방법(쉬운 설명)**: 사진에서 기구(튜브)를 찾은 뒤, 각각의 표시선 안쪽(표적 구간)만 골라 그 부분의 초록색 밝기 중 상위 5% 수준을 대표값으로 삼아 비교했습니다. "
        "즉, 눈으로 봤을 때 밝아 보이는 부분을 과도하게 반영하지 않도록, 여러 픽셀 중 상위 구간의 평균적인 밝기를 사용했다고 이해하면 됩니다. "
        "윗튜브가 기준, 아랫튜브가 검사 대상이며 하단/상단의 비율(Il/Iu)이 임계보다 크면 양성으로 해석합니다.\n"
        "4) 오류/주의 및 해결: 위 노트를 불릿 목록으로, 각 항목에 바로 실행 가능한 해결 방법 포함\n"
        "5) 다음 단계: 증상/성접촉력 고려 진료(산부인과/비뇨의학과), 재촬영 조건, 빠른 내원 기준\n"
        "6) 면책: 본 결과는 참고용 보조 도구이며 확진·치료 지시는 의료진 판단이 필요함\n"
    )

def gemini_answer(user_msg: str, context_ko: str | None = None) -> str:
    # 항상 먼저 Gemini에게 intent를 물어보고
    # intent에 따라 Kakao / CSE / 일반 답변으로 라우팅.
    user_msg = user_msg.strip()
    route = classify_query_with_gemini(user_msg)
    intent = route.get("intent", "general")
    place = (route.get("place") or "").strip()
    spec  = (route.get("specialty") or "").strip()

    # 1) 병원 질의 → Kakao Local 검색
    if intent == "hospital_search":
        if place and spec:
            q = f"{place} {spec}"
        elif place:
            q = f"{place} 병원"
        elif spec:
            q = spec
        else:
            return "어느 지역의 어떤 진료과를 찾는지 조금 더 구체적으로 적어 주세요. (예: '분당 산부인과', '야탑역 비뇨의학과')"

        items = kakao_search_places(q, size=5)
        if not items:
            return "검색 결과가 없습니다. 지명을 더 구체적으로 입력해 주세요. (예: '분당 산부인과', '야탑역 산부인과')"

        lines = []
        for it in items:
            name = it["name"]
            addr = it["address"]
            phone = it["phone"] or "-"
            url = it["url"] or "-"
            lines.append(f"• **{name}** — {addr} / {phone} — {url}")
        return "다음 병원을 참고해 보세요:\n\n" + "\n".join(lines)

    # 2) 의학 최신정보/가이드라인 → CSE + Gemini 요약
    if intent == "med_news" and cse_available():
        sr = google_cse_search(user_msg, num=6)
        if sr:
            summary = "\n".join(
                f"- {i+1}. {r['title']} — {r['snippet']} ({r['link']})"
                for i, r in enumerate(sr)
            )
            prompt = (
                "아래 웹 검색 결과를 근거로 한국어로 간단하고 실용적인 답변을 작성하세요. "
                "정보가 최신이 아닐 수 있음을 한 줄로 언급하고, 확진/처방 지시는 금지합니다.\n\n"
                f"[검색 결과]\n{summary}\n\n"
                "요청:\n"
                "- 핵심 bullet 3–5개와 주의사항 1–2개."
            )
            return gemini_safe_reply(prompt, context_ko or "컨텍스트 없음")

    # 3) 그 외 일반 질의 → LLM-only
    prompt = (
        "자연스럽고 명확한 한국어로 대답하세요. 확진/처방 지시는 금지합니다. "
        "검사결과(컨텍스트)를 기억하고, 임질의 원인/증상/무증상 가능성/예방/다음 단계 등을 사용자 눈높이로 설명하세요.\n"
        f"[사용자 질문]\n{user_msg}\n"
    )
    return gemini_safe_reply(prompt, context_ko or "컨텍스트 없음")

# ================= Streamlit UI =================
st.set_page_config(page_title="스마트폰 기반 임질 진단 시스템", layout="wide")
st.title("스마트폰 기반 임질 진단 시스템")

with st.sidebar:
    st.subheader("설정 (고정값)")
    model_path = st.text_input("YOLOv8 가중치 경로", MODEL_PATH_DEFAULT)
    st.caption("클래스 이름: tube / roi (고정)")
    st.write(f"CONF_MIN = **{CONF_MIN:.2f}**, IOU = {IOU}, IMG_SIZE = {IMG_SIZE}")
    st.write(f"ratio 임계 = **{RATIO_THR}**, ABS_NEG_CUTOFF = **{ABS_NEG_CUTOFF}**")

    try:
        gem_ver = pkg_version("google-generativeai")
        st.caption(f"google-generativeai v{gem_ver}")
    except Exception:
        pass

    if cse_available():
        st.success("검색 모드: Google Custom Search API 사용")
    else:
        st.info("검색 모드: LLM만 (CSE 미설정)")

uploaded = st.file_uploader(
    "기준 샘플(위)과 테스트 샘플(아래)가 함께 보이도록 촬영한 이미지를 업로드하세요. (jpg/png)",
    type=["jpg","jpeg","png"]
)

if uploaded:
    file_bytes = uploaded.read()
    file_bytes_np = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(file_bytes_np, cv2.IMREAD_COLOR)
    img_hash = hashlib.sha1(file_bytes).hexdigest()

    try:
        model = YOLO(str(model_path))
    except Exception as e:
        st.error(f"YOLO 가중치를 불러오지 못했습니다: {e}")
        st.stop()

    Iu, Il, ratio, is_pos, notes, viz_items = detect_pair_and_measure(img_bgr, model)
    viz = overlay_visual(img_bgr, viz_items)
    show_bgr_image_safe(viz, caption="검출 결과 (CONF<0.70 선 숨김)")

    st.subheader("🩺 진단 결과 요약")
    colA, colB, colC = st.columns(3)
    with colA: st.metric("상단 밝기 (G·p95)", f"{Iu:.2f}")
    with colB: st.metric("하단 밝기 (G·p95)", f"{Il:.2f}")
    with colC:
        delta_txt = f"임계 {RATIO_THR}"
        st.metric("비율 Il/Iu", f"{ratio:.3f}" if np.isfinite(ratio) else "N/A", delta=delta_txt)

    if np.isfinite(ratio):
        if is_pos: st.error("조합 판정: **POSITIVE** (양성 가능성 있음)")
        else:      st.success("조합 판정: **NEGATIVE** (음성 가능성 높음)")
    else:
        st.warning("조합 판정 불가")

    for n in notes:
        st.warning("• " + n)

    # --------- Gemini 세션/보고서 ----------
    ratio_fmt = f"{ratio:.3f}" if np.isfinite(ratio) else "nan"
    judge = '양성' if is_pos else ('음성' if np.isfinite(ratio) else '불가')
    context_str = (
        f"- 상단 Iu={Iu:.2f}, 하단 Il={Il:.2f}, ratio={ratio_fmt}\n"
        f"- 판정={judge} (임계={RATIO_THR})"
    )

    # 새 이미지면 새 세션
    if st.session_state.get("last_img_hash") != img_hash:
        st.session_state["last_img_hash"] = img_hash
        st.session_state["gemini_chat"] = None  # 새 컨텍스트로 재생성
        st.session_state["chat_ui"] = []
        st.session_state["gemini_report"] = None

    # 단일 보고서 생성 (한 번만)
    if st.session_state["gemini_report"] is None:
        prompt = make_report_prompt(Iu, Il, ratio, RATIO_THR, is_pos, notes)
        st.session_state["gemini_report"] = gemini_safe_reply(prompt, context_ko=context_str)

    st.markdown("---")
    st.subheader("💡 AI 기반 최종 분석 보고서")
    if st.session_state["gemini_report"]:
        st.markdown(st.session_state["gemini_report"])
    else:
        st.info("요약 보고서를 불러오지 못했습니다.")

    st.markdown("---")
    st.subheader("🤖 AI 챗봇에게 추가 질문하기")
    st.caption("챗봇이 위의 분석 내용을 기억하고 답변합니다.")

    # 기존 대화 표시
    for role, text in st.session_state.get("chat_ui", []):
        (st.chat_message("user") if role=="user" else st.chat_message("assistant")).write(text)

    user_q = st.chat_input("예: '분당 산부인과', '야탑역 산부인과', '임질 무증상도 있어?', '검사 후 뭘 해야 해?'")
    if user_q:
        st.session_state["chat_ui"].append(("user", user_q))
        st.chat_message("user").write(user_q)

        reply = gemini_answer(user_q, context_ko=context_str)
        st.session_state["chat_ui"].append(("assistant", reply))
        st.chat_message("assistant").write(reply)

    # Footer: Powered by Gemini
    _, model_name = _get_gemini_model()
    if model_name:
        st.markdown(
            "<div style='text-align:right; opacity:0.7;'>powered by "
            f"<b>{model_name}</b></div>",
            unsafe_allow_html=True,
        )

else:
    st.info("촬영한 이미지를 업로드하면 자동 분석을 시작합니다.")

