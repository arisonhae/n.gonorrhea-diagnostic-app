# app.py
# ------------------------------------------------------------
# 스마트폰 기반 임질 진단 시스템 (YOLOv8 + G(p95) + Il/Iu ratio)
# - ROI/탐지/비율 계산 기존 방식 유지
# - 보고서에 "어떻게 측정하는지" 설명 포함
# - 병원/의학정보 질의: 우선 카카오맵(위치), 그 외 CSE/LLM 폴백
# - Gemini는 결과 컨텍스트를 기억한 채 대화
# - 이미지 표시는 폭 400px로 축소 표시
# ------------------------------------------------------------

import os, re, json, hashlib
import numpy as np
import cv2
import requests
import streamlit as st
from importlib.metadata import version as pkg_version

# ---------------- YOLO ----------------
try:
    from ultralytics import YOLO
except Exception:
    st.error("ultralytics 패키지가 필요합니다. `pip install ultralytics` 실행 후 재시도하세요.")
    raise

# ===================== 설정/상수 =====================
MODEL_PATH_DEFAULT = "models/new_weights.pt"
CONF_MIN = 0.70
IOU = 0.50
IMG_SIZE = 640

# 고정 임계 (기존 설정 유지)
RATIO_THR = 1.148        # Il/Iu 임계
ABS_NEG_CUTOFF = 221.0   # 상단(음성튜브) 절대 밝기 컷오프

# 시각화
BOX_THICK = 4
FONT_SCALE = 1.15
FONT_THICK = 3
LABEL_ALPHA = 0.65

# BGR
COLOR_TUBE = (0, 255, 0)
COLOR_ROI  = (255, 0, 255)
COLOR_TEXT = (255, 255, 255)

# ===================== 공통 유틸 =====================
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
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    H, W = img.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W - 1, x2); y2 = min(H - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def g_p95(crop_bgr):
    if crop_bgr is None:
        return np.nan
    G = crop_bgr[:, :, 1].astype(np.float32)
    return float(np.percentile(G, 95.0))

def draw_label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICK)
    bg_x1, bg_y1 = x, max(0, y - th - 8)
    bg_x2, bg_y2 = x + tw + 12, y + 4
    overlay = img.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, LABEL_ALPHA, img, 1 - LABEL_ALPHA, 0, img)
    cv2.putText(img, text, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), FONT_THICK + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR_TEXT, FONT_THICK, cv2.LINE_AA)

def draw_box(img, xyxy, color, label=None, show=True):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    if show:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICK)
    if label:
        draw_label(img, label, x1, y1, color)

def show_bgr_image_safe(img_bgr, caption=None, width=400):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption=caption, width=width)

# ===================== 탐지/계산 =====================
def detect_pair_and_measure(img_bgr, model):
    r = model.predict(source=img_bgr, imgsz=IMG_SIZE, conf=CONF_MIN, iou=IOU, verbose=False)[0]
    names = r.names
    inv = {v: k for k, v in names.items()}
    if "tube" not in inv or "roi" not in inv:
        raise RuntimeError(f"모델 클래스에 'tube' 또는 'roi'가 없습니다. names={names}")

    tube_id = inv["tube"]; roi_id = inv["roi"]
    boxes = r.boxes.xyxy.cpu().numpy()
    clses  = r.boxes.cls.cpu().numpy().astype(int)
    confs  = r.boxes.conf.cpu().numpy()

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

    # 사용자 행동지시형 오류/주의
    notes = []
    if len(tubes) == 0 or all(cf < CONF_MIN for cf in tubes_conf):
        notes.append(
            "튜브가 잘 잡히지 않음: 초점이 맞지 않았거나 강한 반사광일 수 있어요. "
            "카메라를 10–15cm 거리에서 정면에 가깝게 두고 렌즈를 닦은 뒤, 상부 조명을 비껴가도록 각도를 약간 바꿔 다시 촬영해주세요."
        )
    if (upper is None or lower is None):
        notes.append(
            "ROI가 하나만 보이거나 안 보임: 용액이 흩어진(splash) 상태일 수 있어요. "
            "튜브를 수직으로 세우고 바닥을 2–3회 톡톡 쳐서 용액이 바닥으로 모이게 한 뒤, 거품이 가라앉으면 재촬영해주세요."
        )
    if np.isfinite(Iu) and Iu >= ABS_NEG_CUTOFF:
        notes.append(
            "상단(기준) 밝기가 비정상적으로 높아요. 상단에는 반드시 음성 대조(NC)를 쓰고, 반사광이 강하면 각도를 조정해주세요."
        )
    if not np.isfinite(ratio):
        notes.append(
            "비율(Il/Iu) 계산이 어려워요. 두 줄(상단/하단)의 측정 구역이 모두 안정적으로 잡혀야 합니다. 위 안내대로 재촬영 후 다시 시도해주세요."
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

# ===================== Gemini =====================
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
    system_prompt = (
        "역할: 임질(Neisseria gonorrhoeae) 체외진단 앱의 한국어 어시스턴트.\n"
        "원칙: 짧고 정확, 일반인 친화 설명. 확진/처방 지시는 금지.\n"
        "핵심 근거: Il/Iu 비율과 고정 임계값.\n\n"
        f"[현재 측정 요약]\n{context_ko}\n"
    )
    try:
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

# 검색 쿼리 정규화 (지역/과목 슬롯 추출)
def gemini_normalize_query(user_q: str) -> dict:
    try:
        model, _ = _get_gemini_model()
        if model is not None:
            prompt = (
                "다음 문장을 병원 검색용으로 구조화하세요.\n"
                "JSON만 출력. 키: region(지명), specialty(진료과), extra(배열), radius_km(숫자). "
                "‘내 위치’ 같은 표현은 region에 넣지 말고 extra에만 넣으세요.\n"
                f"문장: {user_q}\n"
                "예시: {\"region\":\"분당\",\"specialty\":\"산부인과\",\"extra\":[\"근처\"],\"radius_km\":3}"
            )
            resp = model.generate_content(prompt)
            jtxt = resp.text.strip()
            jtxt = re.sub(r"^```json|^```|```$", "", jtxt, flags=re.MULTILINE).strip()
            data = json.loads(jtxt)
            return {
                "region": (data.get("region") or "").strip(),
                "specialty": (data.get("specialty") or "").strip(),
                "extra": data.get("extra") or [],
                "radius_km": float(data.get("radius_km") or 3.0),
            }
    except Exception:
        pass

    # 폴백(간단 전처리)
    stop = r"(근처|주변|가까운|추천|알려줘|좀|최고|베스트|목록|리스트)"
    q = re.sub(stop, " ", user_q)
    q = re.sub(r"\s+", " ", q).strip()
    SPECIALTIES = ["산부인과","비뇨의학과","여성의원","비뇨기과","내과","소아과","피부과","이비인후과","정형외과","가정의학과"]
    specialty = next((s for s in SPECIALTIES if s in q), "")
    region = q.replace(specialty, "").strip()
    return {"region": region, "specialty": specialty, "extra": [], "radius_km": 3.0}

# ===================== 검색(옵션) =====================
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
        items = r.json().get("items", [])
        return [{"title": it.get("title"), "snippet": it.get("snippet"), "link": it.get("link")} for it in items]
    except Exception:
        return []

# Kakao Local 검색 (장소 키워드)
def kakao_search_clinics(region: str, query: str, radius_km: float = 3.0, limit: int = 5) -> tuple[list, str | None]:
    key = st.secrets.get("KAKAO_API_KEY")
    if not key:
        return [], "KAKAO_API_KEY 가 secrets에 없습니다."
    headers = {"Authorization": f"KakaoAK {key}"}

    # 우선 region+query로 키워드 검색
    q = f"{region} {query}".strip()
    try:
        r = requests.get(
            "https://dapi.kakao.com/v2/local/search/keyword.json",
            params={"query": q, "size": limit},
            headers=headers, timeout=6
        )
        if r.status_code != 200:
            return [], f"카카오맵 검색 오류: {r.status_code} {r.text}"
        docs = r.json().get("documents", [])
        out = []
        for d in docs:
            out.append({
                "name": d.get("place_name"),
                "addr": d.get("road_address_name") or d.get("address_name"),
                "phone": d.get("phone"),
                "link": d.get("place_url"),
                "cat": d.get("category_name"),
            })
        return out, None
    except Exception as e:
        return [], f"카카오맵 검색 실패: {e}"

# ===================== 보고서/응답 =====================
def make_report_prompt(Iu, Il, ratio, thr, is_pos, notes):
    ratio_txt = f"{ratio:.3f}" if np.isfinite(ratio) else "계산불가"
    judge = '양성' if is_pos else ('음성' if np.isfinite(ratio) else '판정불가')

    # ★ 측정방법 설명(일반어)
    method_explain = (
        "측정은 다음 순서로 진행됩니다.\n"
        "• 사진에서 두 개의 튜브와 각 튜브의 측정 구역(밝기 읽을 구역)을 자동으로 찾습니다. "
        "이때 신뢰도가 낮은 후보는 자동으로 걸러집니다.\n"
        "• 각 구역의 초록색 밝기 중 상위 5% 수준(G_95)을 대표값으로 사용합니다. "
        "눈부심·노이즈의 영향을 줄이면서 실제 형광 강도를 잘 반영하기 위함입니다.\n"
        "• 하단(테스트) 밝기 Il을 상단(기준) 밝기 Iu로 나눈 비율(Il/Iu)을 계산해 임계값과 비교합니다."
    )

    return (
        "다음 데이터를 바탕으로 환자용 한국어 요약 보고서를 작성하세요.\n"
        "스타일: 제목 1줄 + 간단 근거 + 측정방법(일반어) + 오류/주의(해결 포함) + 다음 단계 + 면책.\n"
        f"- 상단 밝기 Iu={Iu:.2f}, 하단 밝기 Il={Il:.2f}, 비율 Il/Iu={ratio_txt}, 임계={thr:.3f}\n"
        f"- 판정: {judge}\n"
        f"- 참고 노트: {notes}\n\n"
        f"[측정방법]\n{method_explain}\n"
    )

def gemini_answer(chat, user_msg: str) -> str:
    # 1) 병원/위치 질의라면 => Gemini로 슬롯 추출 → Kakao
    if any(k in user_msg for k in ["병원", "산부인과", "비뇨", "여성의원", "클리닉", "의원"]):
        slots = gemini_normalize_query(user_msg)
        region = slots.get("region", "").strip()
        specialty = slots.get("specialty", "").strip() or "산부인과"
        radius_km = float(slots.get("radius_km", 3.0) or 3.0)

        if not region:
            return "검색 결과가 없습니다. 지명을 더 구체적으로 입력해주세요. (예: '분당 산부인과', '야탑역 산부인과')"

        rows, err = kakao_search_clinics(region, specialty, radius_km=radius_km, limit=5)
        if err:
            return f"카카오맵 검색 오류: {err}"
        if not rows:
            return "요청하신 조건으로 찾은 병원 목록이 없습니다."

        lines = []
        for r in rows:
            line = f"• **{r['name']}** — {r['addr'] or '주소 미상'}"
            if r.get("phone"):
                line += f" / {r['phone']}"
            if r.get("link"):
                line += f"\n  {r['link']}"
            lines.append(line)
        return "다음 병원을 참고해 보세요:\n\n" + "\n".join(lines)

    # 2) 의학 최신정보/일반 질문 → CSE가 있으면 요약, 없으면 LLM-only
    if cse_available() and any(k in user_msg for k in ["최신", "가이드라인", "치료법", "내성", "논문", "뉴스"]):
        results = google_cse_search(user_msg, num=6)
        if results:
            brief = "\n".join(f"- {i+1}. {r['title']} — {r['snippet']} ({r['link']})" for i, r in enumerate(results))
            prompt = (
                "아래 웹 검색 결과를 근거로 한국어로 간단하고 실용적인 답변을 작성하세요. "
                "정확하지 않을 수 있음을 한 줄로 고지하고, 확진/처방 지시는 금지합니다.\n\n"
                f"[검색 결과]\n{brief}\n\n"
                "요청: 핵심 bullet 3–5개와 주의사항 1–2개."
            )
            return gemini_generate(chat, prompt)

    # 3) 일반 LLM 답변
    prompt = (
        "자연스럽고 명확한 한국어로 대답하세요. 확진/처방 지시는 금지합니다. "
        "검사결과(컨텍스트)를 기억하고, 임질의 원인/증상/예방/무증상 가능성/다음 단계 등을 사용자 눈높이로 설명하세요.\n"
        f"[사용자 질문]\n{user_msg}\n"
    )
    return gemini_generate(chat, prompt)

# ===================== UI =====================
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
    "기준 샘플(위)와 테스트 샘플(아래)가 함께 보이도록 촬영한 이미지를 업로드하세요. (jpg/png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    file_bytes = uploaded.read()
    file_np = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(file_np, cv2.IMREAD_COLOR)
    img_hash = hashlib.sha1(file_bytes).hexdigest()

    try:
        model = YOLO(str(model_path))
    except Exception as e:
        st.error(f"YOLO 가중치를 불러오지 못했습니다: {e}")
        st.stop()

    Iu, Il, ratio, is_pos, notes, viz_items = detect_pair_and_measure(img_bgr, model)
    viz = overlay_visual(img_bgr, viz_items)
    show_bgr_image_safe(viz, caption="검출 결과 (CONF<0.70 선 숨김)", width=400)

    st.subheader("🩺 진단 결과 요약")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("상단 밝기 (G·p95)", f"{Iu:.2f}")
    with c2: st.metric("하단 밝기 (G·p95)", f"{Il:.2f}")
    with c3:
        st.metric("비율 Il/Iu", f"{ratio:.3f}" if np.isfinite(ratio) else "N/A", delta=f"임계 {RATIO_THR}")

    if np.isfinite(ratio):
        if is_pos: st.error("조합 판정: **POSITIVE** (양성 가능성 있음)")
        else:      st.success("조합 판정: **NEGATIVE** (음성 가능성 높음)")
    else:
        st.warning("조합 판정 불가")

    for n in notes:
        st.warning("• " + n)

    # Gemini 컨텍스트 준비
    ratio_fmt = f"{ratio:.3f}" if np.isfinite(ratio) else "nan"
    judge = '양성' if is_pos else ('음성' if np.isfinite(ratio) else '불가')
    context_str = f"- 상단 Iu={Iu:.2f}, 하단 Il={Il:.2f}, ratio={ratio_fmt}\n- 판정={judge} (임계={RATIO_THR})"

    # 새 이미지면 새 세션
    if st.session_state.get("last_img_hash") != img_hash:
        st.session_state["last_img_hash"] = img_hash
        st.session_state["gemini_chat"] = gemini_start_chat(context_str)
        st.session_state["chat_ui"] = []
        st.session_state["gemini_report"] = None

    # 단일 보고서 생성
    if st.session_state["gemini_report"] is None:
        prompt = make_report_prompt(Iu, Il, ratio, RATIO_THR, is_pos, notes)
        st.session_state["gemini_report"] = gemini_generate(st.session_state["gemini_chat"], prompt)

    st.markdown("---")
    st.subheader("💡 AI 기반 최종 분석 보고서")
    if st.session_state["gemini_report"]:
        st.markdown(st.session_state["gemini_report"])
    else:
        st.info("요약 보고서를 불러오지 못했습니다.")

    st.markdown("---")
    st.subheader("🤖 AI 챗봇에게 추가 질문하기")
    st.caption("위 분석을 기억하고 답변합니다. 위치 질문은 카카오맵으로 실제 병원을 찾아 드립니다.")

    for role, text in st.session_state.get("chat_ui", []):
        if role == "user":
            st.chat_message("user").write(text)
        else:
            st.chat_message("assistant").write(text)

    user_q = st.chat_input("예: '분당 산부인과', '야탑역 산부인과', '임질 무증상도 있어?', '검사 후 뭘 해야 해?'")
    if user_q:
        st.session_state["chat_ui"].append(("user", user_q))
        st.chat_message("user").write(user_q)
        reply = gemini_answer(st.session_state.get("gemini_chat"), user_q)
        st.session_state["chat_ui"].append(("assistant", reply))
        st.chat_message("assistant").write(reply)

    # Footer
    _, model_name = _get_gemini_model()
    if model_name:
        st.markdown(f"<div style='text-align:right; opacity:0.7;'>powered by <b>{model_name}</b></div>", unsafe_allow_html=True)

else:
    st.info("촬영한 이미지를 업로드하면 자동 분석을 시작합니다.")
