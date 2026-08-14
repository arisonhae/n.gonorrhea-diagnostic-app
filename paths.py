# -*- coding: utf-8 -*-
"""
paths.py
프로젝트 공통 경로 설정.

기존 스크립트들은 C:\\n.gonorrhea_diagnostic_app\\... 을 하드코딩하고 있었다.
그 경우 다른 컴퓨터에서는 전부 실패하므로, 경로를 여기 한 곳에서만 정의한다.

기준 위치를 바꾸고 싶으면 환경변수 NGD_ROOT 를 설정한다.
    Windows :  set NGD_ROOT=D:\\my_project
    Linux   :  export NGD_ROOT=/data/my_project

사용 예
    from paths import SOLO_TRAIN, WEIGHTS_PATH, OUTPUT_DIR, ensure_dir
"""

import os
from pathlib import Path

# 이 파일이 저장소 루트에 있다고 가정
ROOT = Path(os.environ.get("NGD_ROOT", Path(__file__).resolve().parent))

# ---- 입력 데이터 ----
DATA_DIR = ROOT / "data"

SOLO_TRAIN = DATA_DIR / "solo" / "train"
SOLO_TEST = DATA_DIR / "solo" / "test"

PAIR_NEGPOS = DATA_DIR / "pair" / "neg_pos"
PAIR_NEGNEG = DATA_DIR / "pair" / "neg_neg"

QC_SPLASH = DATA_DIR / "qc_test" / "error_splash"
QC_BLUR = DATA_DIR / "qc_test" / "error_blur"
QC_LIGHT = DATA_DIR / "qc_test" / "error_light"

SAMPLES = DATA_DIR / "samples"

# ---- 모델 ----
MODELS_DIR = ROOT / "models"
WEIGHTS_PATH = MODELS_DIR / "weights.pt"

# ---- 출력 ----
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = RESULTS_DIR          # 기존 스크립트 호환용 별칭

# ---- 판정 파라미터 (analysis 결과에서 확정된 값) ----
CONF_MIN = 0.70                   # step1
IOU = 0.50
IMG_SIZE = 640
ABS_NEG_CUTOFF = 221.0            # step3
RATIO_THR = 1.148                 # step4


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
            + f"\n\n현재 기준 경로(ROOT): {ROOT}"
            + "\n환경변수 NGD_ROOT 로 기준 경로를 바꿀 수 있습니다."
        )


if __name__ == "__main__":
    # python paths.py 로 현재 설정을 확인할 수 있다.
    print(f"ROOT         : {ROOT}")
    print(f"DATA_DIR     : {DATA_DIR}      exists={DATA_DIR.exists()}")
    print(f"WEIGHTS_PATH : {WEIGHTS_PATH}  exists={WEIGHTS_PATH.exists()}")
    print(f"RESULTS_DIR  : {RESULTS_DIR}   exists={RESULTS_DIR.exists()}")
