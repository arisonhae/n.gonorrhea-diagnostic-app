# -*- coding: utf-8 -*-
"""
paths.py
프로젝트 공통 경로 설정.

기존 스크립트들은 C:\\n.gonorrhea_diagnostic_app\\... 을 하드코딩하고 있었다.
그 경우 다른 컴퓨터에서는 전부 실패하므로, 경로를 여기 한 곳에서만 정의한다.

데이터 위치는 환경변수 NGD_DATA_ROOT 로 지정한다.
지정하지 않으면 저장소 안의 data/ 를 본다.

    Windows :  [Environment]::SetEnvironmentVariable(
                   "NGD_DATA_ROOT", "C:\\n.gonorrhea_diagnostic_app", "User")
    Linux   :  export NGD_DATA_ROOT=/data/ngd

사용 예
    from paths import SOLO_TRAIN, WEIGHTS_PATH, RESULTS_DIR, ensure_dir
"""

import os
from pathlib import Path

# ------------------------------------------------------------------
# 저장소 루트 — 이 파일이 있는 위치
# ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------------
# 데이터 루트
#   NGD_DATA_ROOT 가 있으면 그 아래의 dataset/ 을 쓰고,
#   없으면 저장소 안의 data/ 를 쓴다.
# ------------------------------------------------------------------
_env_root = os.environ.get("NGD_DATA_ROOT")

if _env_root:
    DATA_DIR = Path(_env_root) / "dataset"
else:
    DATA_DIR = REPO_ROOT / "data"

# ---- solo: 튜브 1개, 절댓값 분석용 ----
SOLO_TRAIN = DATA_DIR / "solo" / "train"
SOLO_TEST = DATA_DIR / "solo" / "test"
SOLO_TRAIN_NEG = SOLO_TRAIN / "neg"
SOLO_TRAIN_POS = SOLO_TRAIN / "pos"
SOLO_TEST_NEG = SOLO_TEST / "neg"
SOLO_TEST_POS = SOLO_TEST / "pos"

# ---- pair: 튜브 2개, 비율 분석용 (위=NC, 아래=시료) ----
PAIR_DIR = DATA_DIR / "pair"
PAIR_NEGPOS = PAIR_DIR / "neg_pos"     # 아래가 양성
PAIR_NEGNEG = PAIR_DIR / "neg_neg"     # 아래도 음성 (위양성 확인용)

# ---- qc_test: 촬영 품질 이상 ----
QC_DIR = DATA_DIR / "qc_test"
QC_SPLASH = QC_DIR / "error_splash"
QC_BLUR = QC_DIR / "error_blur"
QC_LIGHT = QC_DIR / "error_light"

# ---- test_all: 기기 호환성 최종 검증 ----
#   주의: 파라미터 선정 단계(step1~4)에서는 절대 사용하지 않는다.
TEST_ALL = DATA_DIR / "test_all"
TEST_GALAXY_NOTE8 = TEST_ALL / "neg_pos_galaxynote8"
TEST_IPHONE13 = TEST_ALL / "neg_pos_iphone13"
TEST_IPHONE13PRO = TEST_ALL / "neg_pos_iphone13pro"

DEVICE_SETS = {
    "galaxy_note8": TEST_GALAXY_NOTE8,
    "iphone13": TEST_IPHONE13,
    "iphone13pro": TEST_IPHONE13PRO,
}

# ---- 저장소 안 예시 이미지 ----
SAMPLES = REPO_ROOT / "data" / "samples"

# ------------------------------------------------------------------
# 모델
# ------------------------------------------------------------------
MODELS_DIR = REPO_ROOT / "models"
WEIGHTS_PATH = MODELS_DIR / "weights.pt"

# 파일명 변경 전 호환 (weights.pt 가 없으면 new_weights.pt 를 찾는다)
if not WEIGHTS_PATH.exists():
    _legacy = MODELS_DIR / "new_weights.pt"
    if _legacy.exists():
        WEIGHTS_PATH = _legacy

# ------------------------------------------------------------------
# 결과 출력
# ------------------------------------------------------------------
RESULTS_DIR = REPO_ROOT / "results"

OUT_STEP1 = RESULTS_DIR / "step1_conf"
OUT_STEP2A = RESULTS_DIR / "step2a_channel"
OUT_STEP2B = RESULTS_DIR / "step2b_separation"
OUT_STEP3 = RESULTS_DIR / "step3_threshold"
OUT_STEP4 = RESULTS_DIR / "step4_ratio"
OUT_ROC = RESULTS_DIR / "roc"
OUT_NEGNEG = RESULTS_DIR / "negneg_check"

# ------------------------------------------------------------------
# 판정 파라미터 — analysis 단계에서 확정된 값
# ------------------------------------------------------------------
CONF_MIN = 0.70          # step1: confidence 스윕으로 선정
IOU = 0.50
IMG_SIZE = 640
ABS_NEG_CUTOFF = 221.0   # step3: 음성 기준선 (G-p95 절댓값)
RATIO_THR = 1.1162        # step4: 양성 판정 비율 (Il/Iu) (사용한 이미지: neg_pos + neg_neg pair (기존 1.148은 neg_pos만 상))


# ------------------------------------------------------------------
# 유틸
# ------------------------------------------------------------------
def ensure_dir(p) -> Path:
    """폴더가 없으면 만들고 Path 를 돌려준다."""
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def check(*paths) -> None:
    """필요한 경로가 실제로 있는지 확인하고, 없으면 알기 쉽게 알려준다."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "다음 경로를 찾을 수 없습니다:\n  "
            + "\n  ".join(missing)
            + f"\n\n현재 데이터 루트: {DATA_DIR}"
            + "\n환경변수 NGD_DATA_ROOT 로 데이터 위치를 지정할 수 있습니다."
        )


if __name__ == "__main__":
    print("=" * 60)
    print("경로 설정 확인")
    print("=" * 60)
    print(f"REPO_ROOT   : {REPO_ROOT}")
    print(f"DATA_DIR    : {DATA_DIR}")
    print(f"  (환경변수 NGD_DATA_ROOT = {os.environ.get('NGD_DATA_ROOT', '미설정')})")
    print(f"  (환경변수 NGD_ROOT      = {os.environ.get('NGD_ROOT', '미설정')})")
    print()

    targets = [
        ("solo/train/neg", SOLO_TRAIN_NEG),
        ("solo/train/pos", SOLO_TRAIN_POS),
        ("solo/test/neg", SOLO_TEST_NEG),
        ("solo/test/pos", SOLO_TEST_POS),
        ("pair/neg_pos", PAIR_NEGPOS),
        ("pair/neg_neg", PAIR_NEGNEG),
        ("qc_test/error_splash", QC_SPLASH),
        ("test_all/galaxynote8", TEST_GALAXY_NOTE8),
        ("test_all/iphone13", TEST_IPHONE13),
        ("test_all/iphone13pro", TEST_IPHONE13PRO),
        ("models/weights", WEIGHTS_PATH),
    ]

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for name, p in targets:
        if p.exists():
            if p.is_dir():
                n = sum(1 for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS)
                print(f"  [OK]   {name:26s} {n:>4d} images")
            else:
                mb = p.stat().st_size / 1024 / 1024
                print(f"  [OK]   {name:26s} {mb:>6.1f} MB  ({p.name})")
        else:
            print(f"  [없음] {name:26s} {p}")