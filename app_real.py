import hashlib
import numpy as np
import cv2
import streamlit as st
from importlib.metadata import version as pkg_version

# ---------------- YOLO ----------------
try:
    from ultralytics import YOLO
except Exception:
    st.error("ultralytics가 필요합니다. `pip install ultralytics` 후 다시 실행하세요.")
    raise

# --------------- 고정 파라미터 ---------------
MODEL_PATH_DEFAULT = r"models\new_weights.pt"
CONF_MIN = 0.70
IOU = 0.50
IMG_SIZE = 640
RATIO_THR = 1.148
ABS_NEG_CUTOFF = 221.0

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

# --------------- 탐지 ---------------
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

    notes = []
    if len(tubes) > 0 and (upper is None or lower is None):
        notes.append("ROI가 하나 이하로 검출되었습니다 (splash 의심).")
    if len(tubes) == 0 and (len(rois) > 0):
        notes.append("tube 미검출 & ROI만 검출되었습니다 (심한 흔들림/빛반사 의심).")
    if np.isfinite(Iu) and Iu >= ABS_NEG_CUTOFF:
        notes.append("상단 튜브의 형광이 비정상적으로 높습니다. 위쪽 튜브에는 NC 시료를 올려주세요.")

    is_positive = (np.isfinite(ratio) and ratio >= RATIO_THR)

    viz_items = dict(
        tubes=[(tb, tcf) for (tb, tcf, _, _) in pairs],
        rois=[(rb, rcf) for (_, _, rb, rcf) in pairs if rb is not None],
        upper=upper, lower=lower
    )

    return Iu, Il, ratio, is_positive, notes, viz_items, (tubes, tubes_conf, rois, rois_conf)

def overlay_visual(img_bgr, viz_items):
    canvas = img_bgr.copy()
    for tb, tcf in viz_items["tubes"]:
        show = (tcf >= CONF_MIN)
        draw_box(canvas, tb, COLOR_TUBE, label=f"CONF {tcf:.2f}", show=show)
    for rb, rcf in viz_items["rois"]:
        show = (rcf >= CONF_MIN)
        draw_box(canvas, rb, COLOR_ROI, label=f"CONF {rcf:.2f}", show=show)
    return canvas

# ---------------- Gemini ----------------
def _gemini_debug_panel():
    try:
        import google.generativeai as genai
        ver = pkg_version("google-generativeai")
        st.sidebar.caption(f"google-generativeai v{ver}")
        try:
            genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", ""))
            names = [m.name for m in genai.list_models()]
            short = [n.split("/")[-1] for n in names]
            st.sidebar.caption("모델 목록: " + ", ".join(short[:12]) + (" ..." if len(short) > 12 else ""))
        except Exception as e:
            st.sidebar.caption(f"모델 조회 실패: {e}")
    except Exception:
        st.sidebar.caption("google-generativeai 패키지를 찾을 수 없습니다.")

def _gemini_start_chat(context_ko: str):
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model_name = st.session_state.get("gemini_model", "gemini-2.5-flash")
        model = genai.GenerativeModel(model_name)
        system_prompt = (
            "역할: 임질(Neisseria gonorrhoeae) 체외진단 앱의 한국어 어시스턴트.\n"
            "원칙: JSON/표/코드/수식 금지."
            "필요 시 일반 가이드와 전문 상담 권유 포함.\n"
            "판정은 ratio(Il/Iu)와 설정 임계값을 기준으로 함.\n\n"
            f"[현재 측정 요약]\n{context_ko}\n"
        )
        chat = model.start_chat(history=[
            {"role": "user", "parts": system_prompt},
            {"role": "model", "parts": "측정 요약을 기억했습니다. 이 결과를 기준으로 답변하겠습니다."}
        ])
        return chat
    except Exception as e:
        st.warning(f"Gemini 세션 초기화 실패: {e}")
        return None

def gemini_summary_via_session(chat):
    if chat is None:
        return None
    try:
        prompt = (
            "임질(N. gonorrhoeae) 의심 여부 안내용 간단 보고서를 작성하세요.\n"
            "1) 한두 문장 요약(양성/음성 판정 + 간단 근거)\n"
            "2) 해석 팁(형광/비율이 의미하는 바, 재촬영 필요 조건)\n"
            "3) 행동 가이드(진료 권고 및 일반적 안내)\n"
            "※ 확정 진단/치료 지시 금지."
        )
        resp = chat.send_message(prompt)
        return getattr(resp, "text", None)
    except Exception as e:
        return f"(Gemini 응답 실패: {e})"

def gemini_send(chat, user_msg: str):
    if chat is None:
        return "(Gemini 비활성화)"
    try:
        allow = st.session_state.get("allow_reco", False)
        region = st.session_state.get("user_region", "").strip()
        if allow:
            policy = (
                "요청 시 인근 병원/의료기관 이름을 제안해도 됩니다. "
                "정확한 최신 정보는 사용자가 직접 확인해야 함을 고지하세요. "
                + (f"가능하면 '{region}' 근처를 고려하세요. " if region else "")
            )
        else:
            policy = "특정 병원/의료기관 추천은 피하세요."
        prompt = (
            "너는 임질 체외진단 앱의 한국어 어시스턴트다. "
            "현재 세션에 저장된 측정 결과를 바탕으로 답한다.\n\n"
            f"[정책]\n{policy}\n\n[사용자 질문]\n{user_msg}\n"
        )
        resp = chat.send_message(prompt)
        return getattr(resp, "text", None) or "(빈 응답)"
    except Exception as e:
        return f"(Gemini 응답 실패: {e})"

# ================= Streamlit UI =================
st.set_page_config(page_title="스마트폰 기반 임질 진단 시스템", layout="wide")
st.title("스마트폰 기반 임질 진단 시스템 (PAIR 전용)")

with st.sidebar:
    st.subheader("설정 (고정값)")
    model_path = st.text_input("YOLOv8 가중치 경로", MODEL_PATH_DEFAULT)
    st.caption("클래스 이름: tube / roi (고정)")
    st.write(f"CONF_MIN = **{CONF_MIN:.2f}**, IOU = {IOU}, IMG_SIZE = {IMG_SIZE}")
    st.write(f"ratio 임계 = **{RATIO_THR}**, ABS_NEG_CUTOFF = **{ABS_NEG_CUTOFF}**")

    # 🔴 수정: Gemini 항상 활성화
    use_gemini = True

    allow_reco = st.toggle("병원/의료기관 '예시' 추천 허용", value=False)
    user_region = st.text_input("지역(선택)", value="", placeholder="예: 분당, 판교, 서현동")
    st.session_state["allow_reco"] = allow_reco
    st.session_state["user_region"] = user_region

    st.markdown("---")
    _gemini_debug_panel()

uploaded = st.file_uploader("PAIR 이미지를 업로드하세요 (jpg/png)", type=["jpg","jpeg","png"])

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

    Iu, Il, ratio, is_pos, notes, viz_items, raw_lists = detect_pair_and_measure(img_bgr, model)
    viz = overlay_visual(img_bgr, viz_items)
    st.image(cv2.cvtColor(viz, cv2.COLOR_BGR2RGB),
             caption="검출 결과 (CONF<0.70 선 숨김)", use_container_width=True)

    st.subheader("🩺 진단 결과 요약")
    colA, colB, colC = st.columns(3)
    with colA: st.metric("상단 평균 밝기(G·p95)", f"{Iu:.2f}")
    with colB: st.metric("하단 평균 밝기(G·p95)", f"{Il:.2f}")
    with colC: st.metric("비율 Il/Iu", f"{ratio:.3f}", delta=f"임계 {RATIO_THR}")

    if np.isfinite(ratio):
        if is_pos: st.error("조합 판정: **POSITIVE** (양성 가능성 있음)")
        else: st.success("조합 판정: **NEGATIVE** (음성 가능성 높음)")
    else:
        st.warning("조합 판정 불가")

    for n in notes:
        st.warning("• " + n)

    context_str = (
        f"[임질 간이 판독]\n"
        f"- 상단 Iu={Iu:.2f}, 하단 Il={Il:.2f}, ratio={ratio:.3f}\n"
        f"- 판정={'양성' if is_pos else '음성' if np.isfinite(ratio) else '불가'}\n"
        + (f"- 메모: {'; '.join(notes)}" if notes else "")
    )

    # 🔴 수정: Gemini 항상 활성화 상태로 실행
    if st.session_state.get("last_img_hash") != img_hash:
        st.session_state["last_img_hash"] = img_hash
        st.session_state["gemini_chat"] = _gemini_start_chat(context_str)
        st.session_state["chat_ui"] = []
        st.session_state["gemini_summary"] = None

    if st.session_state["gemini_summary"] is None:
        st.session_state["gemini_summary"] = gemini_summary_via_session(st.session_state.get("gemini_chat"))

    st.markdown("---")
    st.subheader("🧠 AI 분석 보고서")
    if st.session_state["gemini_summary"]:
        st.markdown(st.session_state["gemini_summary"])
        st.caption(f"Powered by Gemini · {st.session_state.get('gemini_model','?')}")
    else:
        st.info("Gemini 리포트를 생성하지 않았습니다.")

    st.markdown("---")
    st.subheader("💬 AI 챗봇")
    for role, text in st.session_state.get("chat_ui", []):
        (st.chat_message("user") if role=="user" else st.chat_message("assistant")).write(text)
    user_q = st.chat_input("예: 결과 설명해줘 / 병원 추천해줘")
    if user_q:
        st.session_state["chat_ui"].append(("user", user_q))
        st.chat_message("user").write(user_q)
        reply = gemini_send(st.session_state.get("gemini_chat"), user_q)
        st.session_state["chat_ui"].append(("assistant", reply))
        st.chat_message("assistant").write(reply)
else:
    st.info("PAIR 이미지를 업로드하면 자동 분석을 시작합니다.")
