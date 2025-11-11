# app.py
# ------------------------------------------------------------
# 스마트폰 기반 임질 진단 시스템 (YOLOv8 + G(p95) + Il/Iu ratio)
# - Il/Iu, 판정 기준, ROI 측정 방식 유지
# - 보고서 단일화 + 오류/주의(행동지시형) 강화
# - Gemini 대화: 검사결과를 기억하고 답변
# - Google Custom Search API(CSE) 기반 검색(있으면 사용, 없으면 LLM-only)
# - 위치/병원 추천: ★ Kakao Local API 사용 (REST API 키 필요)
# - 하단에 powered by Gemini <model>
# ------------------------------------------------------------

import hashlib
import os
import re
import numpy as np
import cv2
import streamlit as st
from importlib.metadata import version as pkg_version
import requests

# ---------------- YOLO ----------------
try:
    from ultralytics import YOLO
except Exception:
    st.error("ultralytics가 필요합니다. `pip install ultralytics` 후 다시 실행하세요.")
    raise

# ---------------- Gemini ----------------
def _get_gemini_model():
    try:
        import google.generativeai as genai
    except Exception:
        st.warning("google-generativeai 패키지가 필요합니다. `pip install google-generativeai`")
        return None, None
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.warning("GEMINI_API_KEY 가 secrets에 없습니다.")
        return None, None
    try:
        genai.configure(api_key=api_key)
        model_name = st.session_state.get("gemini_model", "gemini-2.5-flash")
        return genai.GenerativeModel(model_name), model_name
    except Exception as e:
        st.warning(f"Gemini 초기화 실패: {e}")
        return None, None

def gemini_start_chat(context_ko: str):
    model, _ = _get_gemini_model()
    if model is None:
        return None
    try:
        system_prompt = (
            "역할: 임질(Neisseria gonorrhoeae) 체외진단 앱의 한국어 어시스턴트.\n"
            "원칙: 짧고 정확, 일반인 친화 설명. 확진/처방 지시는 금지.\n"
            "핵심 근거: Il/Iu 비율과 고정 임계값.\n\n"
            f"[현재 측정 요약]\n{context_ko}\n"
        )
        chat = model.start_chat(history=[
            {"role": "user", "parts": system_prompt},
            {"role": "model", "parts": "측정 요약을 기억했습니다. 바로 질의응답을 시작하겠습니다."}
        ])
        return chat
    except Exception as e:
        st.warning(f"Gemini 세션 생성 실패: {e}")
        return None

def gemini_generate(chat, prompt: str) -> str:
    if chat is None:
        return "(Gemini 비활성화)"
    try:
        resp = chat.send_message(prompt)
        return getattr(resp, "text", None) or "(빈 응답)"
    except Exception as e:
        return f"(Gemini 응답 실패: {e})"

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

# ---------------- Kakao Local (병원/장소 검색) ----------------
def _kakao_headers():
    key = st.secrets.get("KAKAO_API_KEY")
    if not key:
        return None
    return {"Authorization": f"KakaoAK {key}"}

def _clean_hospital_query(user_msg: str) -> str:
    """
    사용자의 병원 요청 문장에서 불필요 단어를 제거하고
    '지명 + 진료과' 형태의 키워드를 만들어 준다.
    """
    q = user_msg.strip()
    # 흔한 불용어 제거
    q = re.sub(r"(추천|근처|가까운|어디|알려줘|찾아줘|검색|병원은|병원좀|병원좀|병원좀요)", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    # 진료과 누락 시 기본값: 산부인과
    if not re.search(r"(산부인과|비뇨|여성의원|성병|성클리닉)", q):
        q = q + " 산부인과"
    return q

def kakao_search_places_markdown(user_msg: str, size: int = 6) -> str:
    """
    카카오 Local '키워드 검색'으로 결과를 받아서
    스트림릿 채팅창에 바로 붙일 수 있는 마크다운 텍스트를 만든다.
    """
    headers = _kakao_headers()
    if headers is None:
        return "※ 카카오맵 API 키가 설정되지 않았습니다. `KAKAO_API_KEY`를 secrets에 추가하세요."
    query = _clean_hospital_query(user_msg)
    try:
        r = requests.get(
            "https://dapi.kakao.com/v2/local/search/keyword.json",
            headers=headers,
            params={"query": query, "size": size},
            timeout=8
        )
        if not r.ok:
            return f"카카오맵 검색 오류: {r.status_code} {r.text[:120]}"
        docs = r.json().get("documents", [])
        if not docs:
            return "검색 결과가 없습니다. 지명을 더 구체적으로 입력해 주세요. (예: '분당 산부인과', '야탑역 산부인과')"

        lines = ["**제공해주신 질의로 찾은 병원 목록입니다.**\n"]
        for d in docs:
            name = d.get("place_name", "")
            addr = d.get("road_address_name") or d.get("address_name") or ""
            phone = d.get("phone") or "전화번호 정보 없음"
            link  = f"https://map.kakao.com/link/map/{d.get('id')}"
            lines.append(f"- 🏥 **{name}**  \n  📍 {addr}  \n  📞 {phone}  \n  🔗 지도: {link}")
        lines.append("\n> 참고: 결과는 최신 정보와 차이가 있을 수 있으니, 방문 전 병원에 직접 문의해 주세요.")
        return "\n".join(lines)
    except Exception as e:
        return f"카카오맵 검색 실패: {e}"

# --------------- 고정 파라미터(변경 금지 영역) ---------------
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

# --------------- 유틸 ---------------
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
    try:
        st.image(img_rgb, caption=caption, width=400)  # 시각화 축소
    except TypeError:
        st.image(img_rgb, caption=caption, width=400)

# --------------- 탐지 (YOLOv8 + G(p95)) ---------------
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

    # ----- 오류/주의 가이드 (사용자 행동 지시형) -----
    notes = []
    # (A) tube 미검출 또는 신뢰도 낮음 → 초점/빛반사
    if len(tubes) == 0 or all(cf < CONF_MIN for cf in tubes_conf):
        notes.append(
            "튜브가 잡히지 않았습니다(또는 검출 신뢰도가 낮음). 카메라 초점/빛반사 가능성이 큽니다. "
            "해결: 카메라를 10–15cm 거리에서 정면에 가깝게 두고 렌즈를 닦은 뒤, "
            "상부 조명이 비치지 않도록 각도를 약간 조정해 재촬영하세요."
        )
    # (B) ROI 한쪽/없음 → splash
    if (upper is None or lower is None):
        notes.append(
            "측정부위(표면 영역)가 하나만 잡히거나 아예 잡히지 않습니다. 내부 용액이 흩어진(splash) 상태일 수 있습니다. "
            "해결: 튜브를 수직으로 세우고 바닥을 2–3회 가볍게 톡톡 쳐서 용액을 바닥으로 모은 후, "
            "거품/흔들림이 가라앉으면 재촬영하세요."
        )
    # (C) NC 밝기 과다
    if np.isfinite(Iu) and Iu >= ABS_NEG_CUTOFF:
        notes.append(
            "상단(기준) 튜브 밝기가 비정상적으로 높습니다. 상단에는 반드시 음성 대조(NC)를 올려 주세요. "
            "반사광이 강하면 각도를 조정해 재촬영하세요."
        )
    # (D) 비율 계산 불가
    if not np.isfinite(ratio):
        notes.append(
            "비율(Il/Iu) 계산이 불가합니다. 두 영역이 모두 안정적으로 잡혀야 합니다. "
            "위 안내대로 재촬영 후 다시 시도하세요."
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

# ---------------- 보고서 / 대화 프롬프트 ----------------
def make_report_prompt(Iu, Il, ratio, thr, is_pos, notes):
    ratio_txt = f"{ratio:.3f}" if np.isfinite(ratio) else "계산불가"
    judge = '양성' if is_pos else ('음성' if np.isfinite(ratio) else '판정불가')
    # ★ 검출/측정 방식 설명을 '일반어'로 명확히 추가
    return (
        "다음 데이터를 바탕으로 환자용 한국어 요약 보고서를 작성하세요.\n"
        "스타일: 제목 1줄 + 간단 근거 + 오류/주의(해결 포함) + 다음 단계 + 면책.\n"
        f"- 상단 밝기 Iu={Iu:.2f}, 하단 밝기 Il={Il:.2f}, 비율 Il/Iu={ratio_txt}, 임계={thr:.3f}\n"
        f"- 판정: {judge}\n"
        f"- 참고 노트: {notes}\n\n"
        "추가 요구사항(일반인 표현):\n"
        "• 카메라가 자동으로 '시험관(튜브) 모양'과 '표면에서 빛을 읽을 위치'를 찾고, "
        "신뢰도가 충분한(대략 0.7 이상으로 확실히 맞다고 판단된) 것만 사용했다고 설명하세요. "
        "전문어(confidence, ROI)는 쓰지 말고 '확실히 맞다고 판단된 항목' 같은 쉬운 표현으로 바꾸세요.\n"
        "• 형광값은 '초록색 밝기 중에서 가장 밝은 상위 5% 영역의 평균'을 대표값으로 썼다고 설명하세요. "
        "이를 'G_95'라고 부르지만 보고서에서는 '초록색 상위 5% 밝기' 같은 쉬운 표현을 사용하세요.\n"
        "• 위아래 두 영역의 값을 비교하여 '하단/상단( Il/Iu ) 비율'이 임계값보다 크면 양성으로 판단한다고 "
        "한 문단으로 이해하기 쉽게 설명하세요.\n\n"
        "구성:\n"
        "1) 한줄 요약: 양성/음성과 간단 근거(Il/Iu와 임계 비교)\n"
        "2) 결과 해석(일반어): Iu/Il/Il·Iu 비율 의미, 이번 숫자의 의미\n"
        "3) 검출·측정 방식: 위의 쉬운 표현 규칙대로 설명(신뢰도 0.7 이상만 사용, 초록색 상위 5% 밝기)\n"
        "4) 오류/주의 및 해결: 위 노트를 불릿 목록으로, 각 항목에 바로 실행 가능한 해결 방법 포함\n"
        "5) 다음 단계: 증상/성접촉력 고려 진료(산부인과/비뇨의학과), 재촬영 조건, 빠른 내원 기준\n"
        "6) 면책: 본 결과는 참고용 보조 도구이며 확진·치료 지시는 의료진 판단이 필요함\n"
    )

def gemini_answer(chat, user_msg: str, location_hint: str | None = None) -> str:
    """
    일반 질문은 Gemini. 병원/위치 질문은 Kakao Local로 직접 검색.
    의학 최신정보 질문 + CSE 설정 시 → CSE 결과 요약 후 응답.
    """
    # 1) 병원/위치 질의 → Kakao 우선
    wants_hospital = any(k in user_msg for k in ["병원", "산부인과", "비뇨", "여성의원", "클리닉"])
    wants_near = any(k in user_msg for k in ["근처", "가까운", "주변", "near"])
    if wants_hospital:
        return kakao_search_places_markdown(user_msg)

    # 2) 의학 최신정보 → CSE 사용
    use_cse = cse_available()
    wants_med_news = any(k in user_msg for k in ["최신", "가이드라인", "치료법", "내성", "논문", "뉴스"])
    if use_cse and wants_med_news:
        sr = google_cse_search(user_msg, num=6)
        if sr:
            summary = "\n".join(f"- {i+1}. {r['title']} — {r['snippet']} ({r['link']})" for i, r in enumerate(sr))
            prompt = (
                "아래 웹 검색 결과를 근거로 한국어로 간단하고 실용적인 답변을 작성하세요. "
                "정확하지 않은 경우 '정보가 최신이 아닐 수 있습니다'를 명시하고, 확진/처방 지시는 금지합니다.\n\n"
                f"[검색 결과]\n{summary}\n\n"
                "요청: 의학 최신정보라면 핵심 bullet 3–5개와 주의사항 1–2개."
            )
            return gemini_generate(chat, prompt)

    # 3) 일반 질문 → LLM-only
    hint = f"\n[지명 힌트] {location_hint}\n" if location_hint else ""
    prompt = (
        "자연스럽고 명확한 한국어로 대답하세요. 확진/처방 지시는 금지.\n"
        "검사결과(컨텍스트)를 기억하고, 일반적인 임질 정보(원인/증상/예방/무증상 가능성/다음 단계)를 "
        "사용자 눈높이로 설명합니다.\n"
        f"[사용자 질문]\n{user_msg}\n{hint}"
    )
    return gemini_generate(chat, prompt)

# ================= Streamlit UI =================
st.set_page_config(page_title="스마트폰 기반 임질 진단 시스템", layout="wide")
st.title("스마트폰 기반 임질 진단 시스템")

with st.sidebar:
    st.subheader("설정 (고정값)")
    model_path = st.text_input("YOLOv8 가중치 경로", MODEL_PATH_DEFAULT)
    st.caption("클래스 이름: tube / roi (고정)")
    st.write(f"CONF_MIN = **{CONF_MIN:.2f}**, IOU = {IOU}, IMG_SIZE = {IMG_SIZE}")
    st.write(f"ratio 임계 = **{RATIO_THR}**, ABS_NEG_CUTOFF = **{ABS_NEG_CUTOFF}**")

    # 버전/키 상태
    try:
        ver = pkg_version("google-generativeai")
        st.caption(f"google-generativeai v{ver}")
    except Exception:
        pass

    if cse_available():
        st.success("검색 모드: Google Custom Search API 사용")
    else:
        st.info("검색 모드: LLM만 (CSE 미설정)")

uploaded = st.file_uploader(
    "기준 샘플(위)와 테스트 샘플(아래)가 함께 보이도록 촬영한 이미지를 업로드하세요. (jpg/png)",
    type=["jpg","jpeg","png"])

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
        st.session_state["gemini_chat"] = gemini_start_chat(context_str)
        st.session_state["chat_ui"] = []
        st.session_state["gemini_report"] = None

    # 단일 보고서 생성 (한 번만)
    if st.session_state["gemini_report"] is None:
        prompt = make_report_prompt(Iu, Il, ratio, RATIO_THR, is_pos, notes)
        st.session_state["gemini_report"] = gemini_generate(st.session_state["gemini_chat"], prompt)

    st.markdown("---")
    st.subheader("💡 AI 기반 최종 분석 보고서")
    if st.session_state["gemini_report"]:\n        st.markdown(st.session_state["gemini_report"])
    else:
        st.info("요약 보고서를 불러오지 못했습니다.")

    st.markdown("---")
    st.subheader("🤖 AI 챗봇에게 추가 질문하기")
    st.caption("챗봇이 위의 분석 내용을 기억하고 답변합니다.")

    for role, text in st.session_state.get("chat_ui", []):
        (st.chat_message("user") if role=="user" else st.chat_message("assistant")).write(text)

    user_q = st.chat_input("예: '분당 산부인과 추천해줘' / '임질 증상이 뭐야?' / '무증상도 있어?' / '검사 후 뭘 해야 해?'")
    if user_q:
        st.session_state["chat_ui"].append(("user", user_q))
        st.chat_message("user").write(user_q)
        reply = gemini_answer(st.session_state.get("gemini_chat"), user_q, None)
        st.session_state["chat_ui"].append(("assistant", reply))
        st.chat_message("assistant").write(reply)

    # Footer: Powered by Gemini
    _, model_name = _get_gemini_model()
    if model_name:
        st.markdown(
            "<div style='text-align:right; opacity:0.7;'>powered by <b>"
            + model_name +
            "</b></div>",
            unsafe_allow_html=True
        )
else:
    st.info("촬영한 이미지를 업로드하면 자동 분석을 시작합니다.")
