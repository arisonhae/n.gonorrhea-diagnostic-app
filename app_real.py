# app_real.py
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
    st.stop()

# --------------- 고정 파라미터 ---------------
# 리눅스/클라우드 호환: 슬래시(/)
MODEL_PATH_DEFAULT = "models/new_weights.pt"

CONF_MIN = 0.70
IOU = 0.50
IMG_SIZE = 640
RATIO_THR = 1.148       # Il/Iu 임계 (고정)
ABS_NEG_CUTOFF = 221.0  # upper(G·p95) 경고 기준

BOX_THICK = 4
FONT_SCALE = 1.15
FONT_THICK = 3
LABEL_ALPHA = 0.65

# 색상 (BGR)
COLOR_TUBE = (0, 255, 0)      # 초록
COLOR_ROI  = (255, 0, 255)    # 분홍(마젠타)
COLOR_TEXT = (255, 255, 255)  # 흰색

# --------------- 유틸 ---------------
def fmt_num(x, fmt="{:.2f}"):
    return fmt.format(x) if (x is not None and np.isfinite(x)) else "N/A"

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
    """G 채널 95퍼센타일 (G·p95) — 변경 금지(요청 사항)"""
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
    # 테두리 효과(검정 외곽선)
    cv2.putText(img, text, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), FONT_THICK + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, COLOR_TEXT, FONT_THICK, cv2.LINE_AA)

def draw_box(img, xyxy, color, label=None, show=True):
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    if show:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICK)
    if label:
        draw_label(img, label, x1, y1, color)

# ---------- 표시용 안전 함수 ----------
def _ensure_uint8_3ch(img):
    """ndarray 이미지를 uint8 3채널 C_CONTIGUOUS로 강제"""
    if img is None or not isinstance(img, np.ndarray):
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim != 3 or img.shape[2] != 3:
        return None
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    return img

def _bgr_to_rgb_safe(img_bgr):
    """BGR → RGB 변환을 안전하게 시도. (연속 메모리 보장)"""
    try:
        out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(out)
    except Exception:
        try:
            out = img_bgr[:, :, ::-1].copy()  # copy()로 연속 메모리 확보(음수 stride 제거)
            return np.ascontiguousarray(out)
        except Exception:
            return None

def _maybe_downscale(img, max_dim=2200):
    """너무 큰 이미지면 표시용으로 다운스케일"""
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img
    scale = max_dim / float(m)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(out)

def show_bgr_image_safe(img_bgr, caption: str):
    """Streamlit에 안전하게 이미지 표시 (RGB 변환 후 표시)"""
    img_bgr = _ensure_uint8_3ch(img_bgr)
    if img_bgr is None:
        st.error("시각화 버퍼가 손상되었거나 형식이 올바르지 않습니다.")
        return
    img_bgr = _maybe_downscale(img_bgr, max_dim=2200)
    img_rgb = _bgr_to_rgb_safe(img_bgr)
    if img_rgb is None or img_rgb.ndim != 3 or img_rgb.shape[2] != 3 or img_rgb.dtype != np.uint8:
        st.error("이미지 색공간 변환/형식 정규화에 실패했습니다.")
        return
    try:
        st.image(img_rgb, caption=caption, use_container_width=True)
    except Exception as e:
        st.error(f"이미지 표시 중 오류: {e}")

# --------------- 탐지 (YOLOv8 + G·p95 유지) ---------------
def detect_pair_and_measure(img_bgr, model):
    """pair 이미지에서 tube/roi 검출 → 위/아래 ROI G·p95 측정 → Il/Iu 비율/판정"""
    r = model.predict(source=img_bgr, imgsz=IMG_SIZE, conf=CONF_MIN, iou=IOU, verbose=False)[0]
    names = r.names
    inv = {v: k for k, v in names.items()} if isinstance(names, dict) else {v: k for k, v in enumerate(names)}
    if "tube" not in inv or "roi" not in inv:
        raise RuntimeError(f"모델 클래스에 'tube' 또는 'roi'가 없습니다. names={names}")

    tube_id = inv["tube"]; roi_id = inv["roi"]

    # YOLO 결과 텐서 → numpy
    boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, "xyxy") else np.zeros((0, 4))
    clses = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, "cls") else np.zeros((0,), dtype=int)
    confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else np.zeros((0,), dtype=float)

    tubes, tubes_conf = [], []
    rois,  rois_conf  = [], []
    for b, c, cf in zip(boxes, clses, confs):
        if c == tube_id:
            tubes.append(to_xyxy(b)); tubes_conf.append(float(cf))
        elif c == roi_id:
            rois.append(to_xyxy(b));  rois_conf.append(float(cf))

    # 각 tube 내부에서 conf 최대인 ROI 1개 선택
    pairs = []
    for ti, tb in enumerate(tubes):
        contained = [(ri, rc) for ri, rc in zip(rois, rois_conf) if inside(ri, tb)]
        if contained:
            contained.sort(key=lambda x: x[1], reverse=True)
            best_ri, best_rc = contained[0]
        else:
            best_ri, best_rc = None, None
        pairs.append((tb, tubes_conf[ti], best_ri, best_rc))

    # y-center로 정렬하여 위/아래 선택
    tri = []
    for (tb, tcf, rb, rcf) in pairs:
        if rb is not None:
            cy = center_y(rb)
            tri.append((cy, tb, tcf, rb, rcf))
    tri.sort(key=lambda x: x[0])   # 위쪽 먼저

    upper, lower = (tri[0] if len(tri) >= 1 else None), (tri[1] if len(tri) >= 2 else None)

    # 측정 (요청: G·p95 방식 유지)
    Iu = Il = np.nan
    if upper:  Iu = g_p95(safe_crop(img_bgr, upper[3]))
    if lower:  Il = g_p95(safe_crop(img_bgr, lower[3]))
    ratio = (Il / Iu) if (np.isfinite(Iu) and Iu > 0) else np.nan

    # 상태/오류 메모
    notes = []
    if len(tubes) > 0 and (upper is None or lower is None):
        notes.append("ROI가 하나 이하로 검출되었습니다 (splash 의심).")
    if len(tubes) == 0 and (len(rois) > 0):
        notes.append("tube 미검출 & ROI만 검출되었습니다 (심한 흔들림/빛반사 의심).")
    if np.isfinite(Iu) and Iu >= ABS_NEG_CUTOFF:
        notes.append("상단 튜브의 형광이 비정상적으로 높습니다. 위쪽 튜브에는 NC 시료를 올려주세요.")

    # 최종 판정(요청: 임계/로직 변경 X)
    is_positive = (np.isfinite(ratio) and ratio >= RATIO_THR)

    viz_items = dict(
        tubes=[(tb, tcf) for (tb, tcf, _, _) in pairs],
        rois=[(rb, rcf) for (_, _, rb, rcf) in pairs if rb is not None],
        upper=upper, lower=lower
    )

    return Iu, Il, ratio, is_positive, notes, viz_items, (tubes, tubes_conf, rois, rois_conf)

def overlay_visual(img_bgr, viz_items):
    """검출 결과 시각화 — 박스/라벨 오버레이"""
    img_bgr = _ensure_uint8_3ch(img_bgr)
    if img_bgr is None:
        return None
    canvas = img_bgr.copy()
    for tb, tcf in viz_items.get("tubes", []):
        show = (tcf >= CONF_MIN)
        draw_box(canvas, tb, COLOR_TUBE, label=f"CONF {tcf:.2f}", show=show)
    for rb, rcf in viz_items.get("rois", []):
        show = (rcf >= CONF_MIN)
        draw_box(canvas, rb, COLOR_ROI, label=f"CONF {rcf:.2f}", show=show)
    return np.ascontiguousarray(canvas)

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
            if short:
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
            "원칙: JSON/표/코드/수식 금지. 필요 시 일반 가이드와 전문 상담 권유 포함.\n"
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
            "1) 한두 문장 요약(양성/음성 판정 + 근거)\n"
            "2) 해석 팁(형광/비율 의미, 재촬영 필요 조건)\n"
            "3) 행동 가이드(진료 권고, 주의사항)\n"
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
                "요청 시 인근 병원/의료기관 이름을 예시로 제안해도 됩니다. "
                "정확한 최신 정보는 사용자가 직접 확인해야 함을 고지하세요. "
                + (f"가능하면 '{region}' 근처를 고려하세요. " if region else "")
            )
        else:
            policy = "특정 병원/의료기관 추천은 피하고, 진료과/절차 위주로 안내하세요."
        prompt = (
            "너는 임질 체외진단 앱의 한국어 어시스턴트다. "
            "현재 세션에 저장된 측정 결과를 바탕으로 답한다. "
            "JSON/표/코드 출력 금지. 간결하고 실용적으로.\n\n"
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

    # Gemini 항상 활성화 (요청)
    use_gemini = True

    allow_reco = st.toggle("병원/의료기관 '예시' 추천 허용", value=False)
    user_region = st.text_input("지역(선택)", value="", placeholder="예: 분당, 판교, 서현동")
    st.session_state["allow_reco"] = allow_reco
    st.session_state["user_region"] = user_region

    st.markdown("---")
    _gemini_debug_panel()

# 파일 업로더
uploaded = st.file_uploader("PAIR 이미지를 업로드하세요 (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded:
    # ---------- 이미지 디코딩 ----------
    file_bytes = uploaded.read()
    file_bytes_np = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(file_bytes_np, cv2.IMREAD_COLOR)

    # ⛔️ 디코딩 실패 가드
    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3:
        st.error("이미지 디코딩에 실패했습니다. JPG/PNG 파일인지 확인해 주세요.")
        st.stop()

    img_hash = hashlib.sha1(file_bytes).hexdigest()

    # ---------- 모델 로드 ----------
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        st.error(f"YOLO 가중치를 불러오지 못했습니다: {e}")
        st.stop()

    # ---------- 분석 ----------
    try:
        Iu, Il, ratio, is_pos, notes, viz_items, raw_lists = detect_pair_and_measure(img_bgr, model)
    except Exception as e:
        st.error(f"검출/측정 중 오류: {e}")
        st.stop()

    # ---------- 시각화 ----------
    viz = overlay_visual(img_bgr, viz_items)
    if viz is None:
        st.warning("시각화 이미지를 생성하지 못했습니다. 검출 결과가 부족할 수 있습니다.")
    else:
        show_bgr_image_safe(
            viz,
            caption="검출 결과(굵은 박스 + CONF 라벨 / conf<0.70는 선 숨김)"
        )

    # ---------- 결과 요약 ----------
    st.subheader("🩺 진단 결과 요약")
    colA, colB, colC = st.columns(3)
    with colA: st.metric("상단 평균 밝기(G·p95)", fmt_num(Iu))
    with colB: st.metric("하단 평균 밝기(G·p95)", fmt_num(Il))
    with colC: st.metric("비율 Il/Iu", fmt_num(ratio, "{:.3f}"), delta=f"임계 {RATIO_THR}")

    if np.isfinite(ratio):
        if is_pos: st.error("조합 판정: **POSITIVE** (양성 가능성 있음)")
        else:      st.success("조합 판정: **NEGATIVE** (음성 가능성 높음)")
    else:
        st.warning("조합 판정 불가: ratio 계산 실패(검출 갯수/품질 확인 필요)")

    if notes:
        for n in notes:
            st.warning("• " + n)

    # ---------- Gemini 컨텍스트 ----------
    context_str = (
        f"[임질 간이 판독]\n"
        f"- 상단 Iu={fmt_num(Iu)}, 하단 Il={fmt_num(Il)}, ratio={fmt_num(ratio, '{:.3f}')}\n"
        f"- 판정={'양성' if is_pos else '음성' if np.isfinite(ratio) else '불가'}\n"
        + (f"- 메모: {'; '.join(notes)}" if notes else "- 메모: 특이사항 없음")
    )

    # ---------- Gemini 보고서 ----------
    st.markdown("---")
    st.subheader("🧠 AI 분석 보고서")

    if use_gemini:
        # 새 이미지면 세션 초기화
        if st.session_state.get("last_img_hash") != img_hash:
            st.session_state["last_img_hash"] = img_hash
            st.session_state["gemini_chat"] = _gemini_start_chat(context_str)
            st.session_state["chat_ui"] = []
            st.session_state["gemini_summary"] = None

        if st.session_state["gemini_summary"] is None:
            st.session_state["gemini_summary"] = gemini_summary_via_session(st.session_state.get("gemini_chat"))

        if st.session_state["gemini_summary"]:
            st.markdown(st.session_state["gemini_summary"])
            st.caption(f"Powered by Gemini · {st.session_state.get('gemini_model','?')}")
        else:
            st.info("Gemini 리포트를 생성하지 않았습니다.")
    else:
        st.caption("Gemini 비활성화 상태입니다. Secrets에 GEMINI_API_KEY를 설정하면 리포트가 생성됩니다.")

    # ---------- Gemini Q&A ----------
    st.markdown("---")
    st.subheader("💬 AI 챗봇")
    if use_gemini:
        for role, text in st.session_state.get("chat_ui", []):
            (st.chat_message("user") if role == "user" else st.chat_message("assistant")).write(text)

        user_q = st.chat_input("예: '지금 결과를 설명해줄래?' / '내 위치 근처의 병원을 추천해줄래?'")
        if user_q:
            st.session_state["chat_ui"].append(("user", user_q))
            st.chat_message("user").write(user_q)
            reply = gemini_send(st.session_state.get("gemini_chat"), user_q)
            st.session_state["chat_ui"].append(("assistant", reply))
            st.chat_message("assistant").write(reply)
    else:
        st.caption("Gemini를 활성화하면 이 영역에서 대화할 수 있습니다.")
else:
    st.info("PAIR 이미지를 업로드하면 자동 분석을 시작합니다.")


