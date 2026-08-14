# 스마트폰 기반 임질 진단 시스템

LAMP-CRISPR 형광 반응 결과를 스마트폰으로 촬영해 자동 판독하는 시스템입니다.
YOLOv8로 튜브와 반응 영역(ROI)을 검출하고, ROI의 형광 강도 비율로 양성/음성을 판정합니다.

성균관대학교 생명공학대학 주관 교내 경진대회 출품작 (2025년 11월).

> 제출 이후 코드를 다시 검토하고 결과를 재현하면서 일부 값과 서술을 정정했습니다.
> 논문·포스터는 제출 당시 기록이므로 수정하지 않았고, 달라진 부분은
> [CHANGELOG.md](CHANGELOG.md)에 정리했습니다.

---

## 판정 원리

한 장의 사진에 **음성 대조군(NC)을 위쪽, 검사 시료를 아래쪽**에 함께 담아 촬영합니다.
두 튜브를 같은 조명·같은 카메라로 동시에 찍으므로, 둘의 비율을 쓰면
기기 차이와 조명 차이가 상쇄됩니다. 절댓값 대신 비율을 쓰는 이유입니다.

실제로 기기별 NC 형광값은 176~209로 최대 19%까지 차이가 났습니다.
절댓값 기준(221.0)으로 판정했다면 iPhone 13 Pro는 전부 음성으로 나왔을 것입니다.

```
사진 입력
   ↓
YOLOv8 검출 (클래스: tube, roi / conf ≥ 0.70)
   ↓
튜브 내부에 완전히 포함된 ROI만 매칭, 튜브당 최고 conf ROI 1개 선택
   ↓
ROI 중심 y좌표로 정렬  →  위 = NC, 아래 = 시료
   ↓
각 ROI의 G채널 95백분위수 계산  →  Iu, Il
   ↓
ratio = Il / Iu
   ↓
ratio ≥ 1.1162  →  양성
```

### 주요 파라미터

| 파라미터 | 값 | 도출 |
|---|---|---|
| `CONF_MIN` | 0.70 | `step1_conf_explorer.py` — 검출 confidence 스윕 |
| 채널 / 지표 | G채널 p95 | `step2a_channel_selection.py` — 7채널 × 2지표 중 Cohen's d 최대 |
| `ABS_NEG_CUTOFF` | 221.0 | `step3_threshold_analysis.py` — train 음성 40장의 99.7 백분위수 |
| `RATIO_THR` | **1.1162** | `step4_ratio_threshold.py` — 221.0 / median(pair NC, n=44) |

G채널을 쓰는 이유는 FAM 형광의 발광 파장이 녹색 영역이기 때문이고,
평균 대신 95백분위수를 쓰는 이유는 ROI 안에 어두운 배경 픽셀이 섞여 있어
평균이 신호를 희석시키기 때문입니다.

`RATIO_THR`은 논문 제출본의 1.148과 다릅니다. 제출 이후 추가 촬영한
음성-음성 쌍 데이터를 기준 분포에 포함해 재계산한 값입니다. 자세한 근거는
[CHANGELOG.md](CHANGELOG.md)에 있습니다.

---

## 성능

표본이 작아 신뢰구간이 넓습니다. 점추정값만 보면 실제 성능을 과대평가하게 됩니다.

| 데이터 | n | 지표 | 값 | 95% CI |
|---|---|---|---|---|
| solo test | 30 | 정확도 (절대 기준) | 96.7% | 83.3 – 99.4% |
| pair 전체 | 44 | 정확도 (비율 기준) | 88.6% | 76.0 – 95.0% |
| test_all | 13 | 아래 튜브 정확도 | 92.3% | 66.7 – 98.6% |

**위양성률 16.7%** — 음성-음성 쌍 24장 중 4장이 양성으로 오판됩니다.
Youden's J로 임계값을 최적화해도 같은 값이 나오므로, 임계값 조정으로는
줄일 수 없는 신호 분포 자체의 겹침입니다.

---

## 저장소 구조

```
├── app_real.py                 Streamlit 웹앱 (배포본)
├── paths.py                    공통 경로·파라미터 설정
├── models/weights.pt           YOLOv8s 학습 가중치
│
├── analysis/                   분석 파이프라인
│   ├── step1_conf_explorer.py            검출 confidence 스윕
│   ├── step2a_channel_selection.py       14개 조합 중 G_p95 선정
│   ├── step2b_signal_separation.py       양성/음성 분리 검정
│   ├── step3_threshold_analysis.py       절대 음성 기준선
│   ├── step4_ratio_threshold.py          비율 판정 임계값
│   │
│   ├── roc_analysis.py                   ROC / AUC
│   └── negneg_false_positive_check.py    음성-음성 쌍 위양성 확인
│
├── data/README.md              데이터셋 구조 설명
├── results/                    분석 결과 (csv, json, 그래프)
├── requirements.txt            웹앱 실행용
└── requirements-analysis.txt   분석 스크립트 추가 패키지
```

`step1`~`step4`는 순서대로 이어지는 파이프라인이고, 아래 두 개는
파이프라인 밖에서 결과를 평가하는 도구입니다. 후자는 논문 제출 이후
추가로 수행한 검증입니다.

원본 이미지는 용량 문제로 저장소에 포함하지 않았습니다.
구조와 촬영 조건은 [data/README.md](data/README.md)를 참고하세요.

---

## 실행

### 웹앱

```bash
pip install -r requirements.txt
streamlit run app_real.py
```

Gemini 리포트 기능을 쓰려면 `.streamlit/secrets.toml`에 API 키가 필요합니다.

```toml
GEMINI_API_KEY = "..."
```

### 분석 파이프라인

```bash
pip install -r requirements.txt
pip install -r requirements-analysis.txt

# 데이터 위치 지정 (저장소 밖에 있는 경우)
export NGD_DATA_ROOT=/path/to/data     # Windows: set NGD_DATA_ROOT=...

python paths.py                        # 경로 설정 확인
python analysis/step3_threshold_analysis.py
python analysis/step4_ratio_threshold.py
```

`python paths.py`를 실행하면 각 데이터 폴더의 이미지 개수와 모델 파일이
제대로 잡히는지 확인할 수 있습니다.

검출 오버레이 이미지는 기본적으로 저장하지 않습니다. 필요하면 `--save_viz`를
붙이세요. 다만 이미지 수만큼 파일이 생성됩니다.

---

## 모델

- 아키텍처: **YOLOv8s** (YOLOv8n이 아닙니다 — 논문 기재 오류)
- 클래스: `roi`, `tube` — **2개** (논문 기재 3개와 다릅니다)
- 학습: 300 epoch 설정, patience 90으로 **213 epoch에서 조기 종료** (best는 123 epoch)
- 입력 크기: 640, seed 0
- 데이터셋 관리: Roboflow

검출 모델은 양성/음성을 판별하지 않습니다. 튜브와 ROI의 위치만 찾고,
판정은 전적으로 ROI의 형광값 비율로 이루어집니다.

---

## 한계

학부생 팀이 약 3개월 동안 수행한 프로젝트이며, 아래는 이후 코드와 결과를
다시 검토하며 확인한 한계입니다.

### 검증 범위

**검출 한계(LOD)를 확립하지 못했습니다.** 농도 의존적 형광 패턴이 명확히
나타나지 않아, 특정 농도에서 검출되었다는 단일 관찰만 있습니다.

**특이도 검증에 인간 gDNA만 사용했습니다.** 표적으로 삼은 `porA` pseudogene은
본래 임질균을 다른 *Neisseria* 종과 구별하기 위한 것이므로, 근연종
(*N. meningitidis* 등)에 대한 교차 반응 확인이 필요하나 수행하지 못했습니다.

**실제 임상 검체가 아닌 합성 표적 DNA로 검증했습니다.** 진단 도구가 아니라
개념 검증(proof-of-concept) 수준입니다.

**촬영 조건 최적화 실험 일부는 조건당 n=1로 수행되어**, 통계적 비교의 근거가
되기 어렵습니다.

### 통계적 한계

**절대 기준선이 단일 극단값에 의존합니다.** cutoff 221.0은 train 음성 40장의
99.7 백분위수인데, n=40에서 이 값은 최댓값과 같아집니다. 실제로 train 음성의
최댓값이 정확히 221.0입니다.

**비율 임계값의 신뢰구간이 넓습니다.** 부트스트랩 95% CI가 [1.09, 1.16]으로,
이 범위 안에서는 어느 값을 택하든 성능 차이가 통계적으로 유의하지 않습니다.

**모든 정확도 수치의 신뢰구간이 넓습니다.** 위 성능 표를 참고하세요.

### 데이터 분할

**YOLO 학습 데이터를 이미지 단위로 분할했습니다.** 같은 튜브를 여러 각도·조건으로
촬영한 이미지가 train과 test에 나뉘어 들어갔을 가능성이 있으며, 이 경우
검출 성능이 낙관적으로 평가됩니다. 시료 단위 재분할이 필요합니다.

### 시스템 견고성

**판정 마진이 기기 간 편차보다 작습니다.** 양성 판정 마진은 약 11.6%인데
기기 간 NC 형광값 차이는 최대 19%입니다. 비율 정규화로 상당 부분 상쇄되지만,
경계 근처 시료는 기기에 따라 판정이 달라질 수 있습니다.

**위쪽 튜브의 QC 경고는 여전히 절대 기준을 씁니다.** Galaxy Note 8의 NC 평균이
209.2로 경고 기준(221.0)에 근접해, 기기에 따라 오작동할 수 있습니다.

**튜브를 뒤집어 놓은 경우** 경고는 표시되지만 판정은 그대로 출력됩니다.
안전상 판정 불가로 처리하는 것이 맞습니다.

**위쪽 튜브 판정이 검증되지 않았습니다.** 검증 데이터에 위쪽이 양성인
이미지가 없어, 해당 경로의 정확도를 확인할 수 없습니다.

### 검출 실패

기기 호환성 검증 이미지 22장 중 9장이 ROI 검출에 실패했습니다.
그중 4장은 의도적으로 촬영한 오류 이미지지만, 나머지는 정상 이미지입니다.
검출 confidence 임계값이 높아서인지 확인이 필요합니다.

---

## 문서

- [CHANGELOG.md](CHANGELOG.md) — 논문 제출 이후 변경 사항과 근거
- [data/README.md](data/README.md) — 데이터셋 구조, 촬영 조건, 라벨 규칙
- [results/README.md](results/README.md) — 분석 결과 폴더 구조