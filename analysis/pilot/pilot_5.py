# pilot_5_means.py
# 목적: 기기 간 평균 차이 + 배경 레벨 + (Pos-Neg) 차이만 산출 (의사결정용 슬림 버전)

import json
from pathlib import Path
import math

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ====== 한글 폰트(가능 시) ======
try:
    plt.rcParams["font.family"] = "Noto Serif KR"
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

# ================== 환경에 맞게 경로 수정 ==================
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from paths import OUT_PILOT
# 이 스크립트는 분석기가 아니라 결과 뷰어다. 촬영 원본이 아니라
# 산출물 폴더를 읽는다. summary_means.json 을 만드는 코드는
# 저장소에 남아 있지 않다 (analysis/pilot/README.md 참고).
DEFAULT_BASE = str(OUT_PILOT / "pilot_5_device_submission")
# =========================================================

st.set_page_config(page_title="파일럿 평균 분석 뷰어", layout="wide")
st.title("📊 파일럿 5 (평균 중심): 기기별 비교")

# ====== 경로 입력 & 로드 ======
base_dir = st.text_input("summary_means.json 이 있는 폴더", value=DEFAULT_BASE)
analysis_dir = Path(base_dir)
summary_path = analysis_dir / "summary_means.json"

if not summary_path.exists():
    st.warning(f"summary_means.json을 찾을 수 없습니다: {summary_path}")
    st.stop()

with open(summary_path, "r", encoding="utf-8") as f:
    data = json.load(f)

devices = data.get("devices", []) or []
means = data.get("means", {}) or {}
fails = data.get("fails", []) or []
preview_dir = Path(data.get("preview_dir", str(analysis_dir / "previews_means")))

# ====== 실패 로그 ======
if fails:
    with st.expander("❌ 처리 실패 이미지 목록", expanded=False):
        st.dataframe(pd.DataFrame(fails), use_container_width=True)

st.markdown("---")

# ====== 기기별 요약 테이블 ======
st.subheader("📌 기기별 요약(평균 중심)")

rows = []
for d in devices:
    s = means.get(d, {}) or {}
    pos_mean = s.get("pos_mean")
    neg_mean = s.get("neg_mean")

    # ▶ 상대 차이(%)는 Neg 기준으로 재계산: |Pos-Neg| / Neg * 100
    pos_neg_diff_pct = None
    if isinstance(pos_mean, (int, float)) and isinstance(neg_mean, (int, float)) and neg_mean not in (None, 0):
        pos_neg_diff_pct = abs(pos_mean - neg_mean) / abs(neg_mean) * 100.0

    rows.append({
        "기기": d,
        "Pos 개수": s.get("pos_count"),
        "Neg 개수": s.get("neg_count"),
        "Pos 평균": round(pos_mean, 2) if isinstance(pos_mean, (int, float)) else None,
        "Neg 평균": round(neg_mean, 2) if isinstance(neg_mean, (int, float)) else None,
        "Pos-Neg 차이(절대)": round(abs(pos_mean - neg_mean), 2) if isinstance(pos_mean, (int, float)) and isinstance(neg_mean, (int, float)) else None,
        "Pos-Neg 차이(%)": round(pos_neg_diff_pct, 1) if isinstance(pos_neg_diff_pct, (int, float)) else None,
        "배경 평균(Pos)": round(s["bg_mean_pos"], 2) if isinstance(s.get("bg_mean_pos"), (int, float)) else None,
        "배경 평균(Neg)": round(s["bg_mean_neg"], 2) if isinstance(s.get("bg_mean_neg"), (int, float)) else None,
        "배경 평균(전체)": round(s["bg_mean_overall"], 2) if isinstance(s.get("bg_mean_overall"), (int, float)) else None,
        "SNR(Pos)": round(s["snr_pos_mean"], 2) if isinstance(s.get("snr_pos_mean"), (int, float)) else None,
        "SNR(Neg)": round(s["snr_neg_mean"], 2) if isinstance(s.get("snr_neg_mean"), (int, float)) else None,
    })

if rows:
    df_stats = pd.DataFrame(rows)
    st.dataframe(df_stats, use_container_width=True)
else:
    st.info("요약 데이터가 없습니다.")

st.caption("※ 상대 차이(%) = |Pos_mean − Neg_mean| / Neg_mean × 100 (Neg 기준)")

st.markdown("---")

# ====== 기기 간 평균 차이(%) & 결론 (iphone13pro 기준 한정) ======
st.subheader("🔍 기기 간 평균 차이(%) — 기준: iphone13pro")

reference = "iphone13pro"
comp_df = None
max_diff = None

if reference not in devices:
    st.warning(f"참조 기기 '{reference}'가 devices 목록에 없습니다. summary_means.json의 'devices'를 확인하세요.")
else:
    comp_rows = []
    diffs = []

    ref = means.get(reference, {}) or {}
    ref_pos = ref.get("pos_mean")
    ref_neg = ref.get("neg_mean")

    # 시각화용 리스트
    labels, pos_diff_list, neg_diff_list = [], [], []

    for d in devices:
        if d == reference:
            continue
        m = means.get(d, {}) or {}
        pos = m.get("pos_mean")
        neg = m.get("neg_mean")

        # ▶ 기기간 차이 정의: |Ref - Dev| / Ref * 100 (분모를 iphone13pro로 고정)
        pos_diff_pct = None
        neg_diff_pct = None
        if isinstance(ref_pos, (int, float)) and isinstance(pos, (int, float)) and ref_pos not in (None, 0):
            pos_diff_pct = abs(ref_pos - pos) / abs(ref_pos) * 100.0
        if isinstance(ref_neg, (int, float)) and isinstance(neg, (int, float)) and ref_neg not in (None, 0):
            neg_diff_pct = abs(ref_neg - neg) / abs(ref_neg) * 100.0

        comp_rows.append({
            "비교": f"{reference} vs {d}",
            "Pos 차이(%)": round(pos_diff_pct, 1) if isinstance(pos_diff_pct, (int, float)) else None,
            "Neg 차이(%)": round(neg_diff_pct, 1) if isinstance(neg_diff_pct, (int, float)) else None,
        })

        # 그래프 데이터 축적
        labels.append(d)
        pos_diff_list.append(pos_diff_pct if isinstance(pos_diff_pct, (int, float)) else float("nan"))
        neg_diff_list.append(neg_diff_pct if isinstance(neg_diff_pct, (int, float)) else float("nan"))

        if isinstance(pos_diff_pct, (int, float)):
            diffs.append(pos_diff_pct)
        if isinstance(neg_diff_pct, (int, float)):
            diffs.append(neg_diff_pct)

    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df, use_container_width=True)
    max_diff = max(diffs) if diffs else None

    # === 시각화: 막대 그래프(두 개) ===
    st.markdown("##### 📊 기기간 차이 시각화")
    c1, c2 = st.columns(2)

    with c1:
        fig_pos, ax_pos = plt.subplots(figsize=(5, 3.8))
        ax_pos.bar(
            labels,
            [0 if (v is None or math.isnan(v)) else v for v in pos_diff_list],
            color="tab:orange",
            width=0.5,
            edgecolor="black"
        )
        ax_pos.set_title("Pos 차이(%) — 기준: iphone13pro")
        ax_pos.set_ylabel("차이율(%)")
        ax_pos.tick_params(axis="x", rotation=45)
        st.pyplot(fig_pos)

    with c2:
        fig_neg, ax_neg = plt.subplots(figsize=(5, 3.8))
        ax_neg.bar(
            labels,
            [0 if (v is None or math.isnan(v)) else v for v in neg_diff_list],
            color="tab:cyan",
            width=0.5,
            edgecolor="black"
        )
        ax_neg.set_title("Neg 차이(%) — 기준: iphone13pro")
        ax_neg.set_ylabel("차이율(%)")
        ax_neg.tick_params(axis="x", rotation=45)
        st.pyplot(fig_neg)

    ax_pos.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_neg.grid(True, axis="y", linestyle="--", alpha=0.4)

    # === 시각화: 산점도 (Neg 평균 vs Pos 평균) ===
    st.markdown("##### 🔎 기기별 Pos/Neg 평균 분포(참고)")
    fig_sc, ax_sc = plt.subplots(figsize=(5.8, 4.2))
    for d in devices:
        m = means.get(d, {}) or {}
        x = m.get("neg_mean")
        y = m.get("pos_mean")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if d == reference:
                ax_sc.scatter(x, y, s=80, marker="*", label=d)  # ref는 별표
            else:
                ax_sc.scatter(x, y, s=40, label=d)
            ax_sc.annotate(d, (x, y), xytext=(5, 3), textcoords="offset points", fontsize=9)
    ax_sc.set_xlabel("Neg 평균")
    ax_sc.set_ylabel("Pos 평균")
    ax_sc.set_title("기기별 Pos/Neg 평균 산점도")
    ax_sc.grid(True, alpha=0.3)
    st.pyplot(fig_sc)

# 상단 결론 배너(평균 차이 기준)
st.markdown("### 🎯 결론(의사결정 가이드)")
if max_diff is None:
    st.info("비교 가능한 기기간 평균 차이가 없습니다. (데이터 부족)")
else:
    if max_diff < 10:
        st.success("✅ **한 기종으로 촬영해도 충분**합니다. (기기 간 평균 차이 < 10%)")
        st.info("가능하면 노출/화이트밸런스를 고정하여 촬영하면 후처리 부담이 더 줄어듭니다.")
    elif max_diff < 20:
        st.warning("⚠️ **한 기종 촬영 + 간단 정규화 권장** (평균 차이 10–20%)")
        st.info("예: 기기별 Neg 또는 배경 평균으로 스케일링하여 정규화.")
    else:
        st.error("❌ **기기 혼용 시 정규화 필수** (평균 차이 ≥ 20%)")
        st.info("가능하면 한 기종으로 통일하거나, 기기별 보정 파이프라인을 분리하세요.")

st.markdown("---")

# ====== 그래프: 평균 비교 ======
if devices and means:
    st.subheader("📈 평균 비교 그래프")
    dev_order = [d for d in devices if d in means]  # 원래 순서 유지

    def safe_list(key):
        vals = []
        for d in dev_order:
            v = means[d].get(key, None)
            vals.append(v if isinstance(v, (int, float)) and not math.isnan(v) else None)
        return vals

    pos_means = safe_list("pos_mean")
    neg_means = safe_list("neg_mean")
    bg_overall = safe_list("bg_mean_overall")

    col1, col2, col3 = st.columns(3)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.bar(dev_order, [v if v is not None else 0 for v in pos_means])
        ax1.set_title("Positive 평균 신호")
        ax1.set_ylabel("신호 강도")
        ax1.tick_params(axis='x', rotation=45)
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.bar(dev_order, [v if v is not None else 0 for v in neg_means])
        ax2.set_title("Negative 평균 신호")
        ax2.set_ylabel("신호 강도")
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        ax3.bar(dev_order, [v if v is not None else 0 for v in bg_overall])
        ax3.set_title("배경 평균(전체)")
        ax3.set_ylabel("신호 강도")
        ax3.tick_params(axis='x', rotation=45)
        st.pyplot(fig3)

st.markdown("---")

# ====== 정규화 팩터(선택) ======
st.subheader("🧮 정규화 팩터(참고)")
if devices and means:
    # 참조 기기 선택 (기본: iphone13pro)
    default_index = devices.index("iphone13pro") if "iphone13pro" in devices else 0
    ref_device = st.selectbox("참조 기기(Neg 평균 기준, 배율=Ref_Neg / Dev_Neg)", devices, index=default_index)
    ref_neg = means.get(ref_device, {}).get("neg_mean", None)

    norm_rows = []
    for d in devices:
        neg = means.get(d, {}).get("neg_mean", None)
        factor = None
        if isinstance(ref_neg, (int, float)) and isinstance(neg, (int, float)) and neg not in (None, 0):
            factor = ref_neg / neg
        norm_rows.append({
            "기기": d,
            "Neg 평균": round(neg, 2) if isinstance(neg, (int, float)) else None,
            f"정규화 팩터 (→{ref_device})": round(factor, 4) if isinstance(factor, (int, float)) else None
        })
    st.dataframe(pd.DataFrame(norm_rows), use_container_width=True)
    st.caption("※ 혼용 촬영 시, 각 기기의 값을 위 배율로 곱해 맞추면 평균 레벨이 정렬됩니다.")

st.markdown("---")

# ====== 이미지 프리뷰 ======
st.subheader("🖼️ 오버레이 프리뷰")
if not preview_dir.exists():
    st.info("프리뷰 디렉토리를 찾을 수 없습니다.")
else:
    if devices:
        tabs = st.tabs(devices)
        for d, tab in zip(devices, tabs):
            with tab:
                dev_dir = preview_dir / d
                pos_dir = dev_dir / "positive"
                neg_dir = dev_dir / "negative"
                sub1, sub2 = st.tabs(["Positive", "Negative"])

                def show_grid(img_dir: Path):
                    if not img_dir.exists():
                        st.info("폴더 없음")
                        return
                    imgs = sorted(img_dir.glob("*_overlay.jpg"))
                    if not imgs:
                        st.info("오버레이 없음")
                        return
                    ncols = 2
                    for i in range(0, len(imgs), ncols):
                        cols = st.columns(ncols)
                        for j in range(ncols):
                            k = i + j
                            if k < len(imgs):
                                img = Image.open(imgs[k])
                                cols[j].image(img, caption=imgs[k].name, use_container_width=True)

                with sub1:
                    show_grid(pos_dir)
                with sub2:
                    show_grid(neg_dir)

st.caption(f"요약 파일: {summary_path}")
st.caption(f"생성 시각: {data.get('generated_at','N/A')}")