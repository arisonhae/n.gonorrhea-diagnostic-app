# GonoCheck

[![Code DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22090179.svg)](https://doi.org/10.5281/zenodo.22090179)
[![Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22090185.svg)](https://doi.org/10.5281/zenodo.22090185)

스마트폰 기반 임질 진단 시스템

**[앱 실행하기](https://ngonorrhea-diagnostic-app-tfpfnjgw8ppcs4fqwbrvar.streamlit.app/)** · [샘플 이미지로 바로 테스트](data/samples/)

> 무료 호스팅이라 12시간 동안 접속이 없으면 대기 상태로 전환된다.
> 대기 화면이 뜨면 "Yes, get this app back up!" 을 누르고 30초쯤 기다리면 된다.

LAMP-CRISPR 형광 반응 결과를 스마트폰으로 촬영해 자동 판독한다.
YOLOv8로 튜브와 반응 영역(ROI)을 검출하고, ROI의 형광 강도 비율로 양성/음성을 판정한다.

성균관대학교 생명공학대학 주관 교내 경진대회 출품작 (2025년 11월).

> 제출 이후 코드를 다시 검토하고 결과를 재현하면서 일부 값과 서술을 정정했다.
> 논문·포스터는 제출 당시 기록이므로 수정하지 않았고, 달라진 부분은
> [CHANGELOG.md](CHANGELOG.md)에 정리했다.

---

## 판정 원리

한 장의 사진에 **음성 대조군(NC)을 위쪽, 검사 시료를 아래쪽**에 함께 담아 촬영한다.
두 튜브를 같은 조명·같은 카메라로 동시에 찍으므로, 둘의 비율을 쓰면
기기 차이와 조명 차이가 상쇄된다. 절댓값 대신 비율을 쓰는 이유다.

실제로 기기별 NC 형광값은 176~209로 최대 19%까지 차이가 났다.
절댓값 기준(221.0)으로 판정했다면 iPhone 13 Pro는 전부 음성으로 나왔을 것이다.

```
사진 입력
   ↓
YOLOv8 검출 (클래스: roi, tube / conf ≥ 0.70)
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
평균이 신호를 희석시키기 때문이다.

`RATIO_THR`은 논문 제출본의 1.148과 다르다. 제출 이후 추가 촬영한
음성-음성 쌍 데이터를 기준 분포에 포함해 재계산한 값이다. 자세한 근거는
[CHANGELOG.md](CHANGELOG.md)에 있다.

---

## 성능

표본이 작아 신뢰구간이 넓다. 점추정값만 보면 실제 성능을 과대평가하게 된다.

| 데이터 | n | 지표 | 값 | 95% CI |
|---|---|---|---|---|
| solo test | 30 | 정확도 (절대 기준) | 96.7% | 83.3 – 99.4% |
| pair 전체 | 44 | 정확도 (비율 기준) | 88.6% | 76.0 – 95.0% |
| test_all | 13 | 아래 튜브 정확도 | 92.3% | 66.7 – 98.6% |

임계값에 의존하지 않는 지표로는 AUC가 있다.

| 데이터 | AUC | 95% CI |
|---|---|---|
| solo train (G_p95 절댓값) | 0.9994 | 0.9972 – 1.0000 |
| pair (ratio) | 0.9646 | 0.9027 – 0.9979 |

절댓값으로는 거의 완전히 갈리지만 비율로 바꾸면 성능이 떨어진다.
비율은 기기 차이를 상쇄해 주는 대신 NC 튜브의 변동을 새로 끌어들이기 때문이다.
기기별 편차가 19%에 이르므로 그 대가를 치를 만하지만, 대가가 있다는 점은
분명히 해 둘 필요가 있다.

**위양성률 16.7%** (95% CI 6.7 – 35.9%) — 음성-음성 쌍 24장 중 4장이 양성으로
오판된다. Youden's J로 임계값을 최적화해도 같은 값이 나오므로, 임계값 조정으로는
줄일 수 없는 신호 분포 자체의 겹침이다.

음성-음성 쌍은 26장을 촬영했으나 2장은 ROI가 하나만 검출되어 비율을 계산할 수
없었다. 위 표의 pair 44장은 `neg_pos` 20장과 `neg_neg` 24장을 합한 것이다.

---

## 저장소 구조

```
├── app.py                      Streamlit 웹앱 (배포본)
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
│   ├── negneg_false_positive_check.py    음성-음성 쌍 위양성 확인
│   └── pilot/                            촬영 조건 최적화 실험 (pilot_1~6)
│
├── data/samples/               예시 이미지 9장 + 기대 결과
├── results/                    분석 결과 (csv, json, 그래프)
├── requirements.txt            웹앱 실행용
└── requirements-analysis.txt   분석 스크립트 추가 패키지
```

`step1`~`step4`는 순서대로 이어지는 파이프라인이고, `roc_analysis` 와
`negneg_false_positive_check` 는 파이프라인 밖에서 결과를 평가하는 도구다.
후자는 논문 제출 이후 추가로 수행한 검증이다.

원본 이미지는 용량 문제로 저장소에 포함하지 않았다.
구조와 촬영 조건은 [data/README.md](data/README.md)를 참고하면 된다.

전체 이미지 데이터(2.7 GB)는 Zenodo 에 있다:
[10.5281/zenodo.22090185](https://doi.org/10.5281/zenodo.22090185)

---

## 실행

### 웹앱

```bash
pip install -r requirements.txt
streamlit run app.py
```

Gemini 설명 기능을 쓰려면 `.streamlit/secrets.toml`에 API 키가 필요하다.
없어도 앱은 실행되며, 설명과 병원 검색 기능만 비활성화된다.

```toml
GEMINI_API_KEY = "..."
```

`data/samples/` 의 이미지를 올리면 각 케이스의 동작을 바로 확인할 수 있다.

### 분석 파이프라인

```bash
pip install -r requirements.txt
pip install -r requirements-analysis.txt

# 데이터 위치 지정 (저장소 밖에 있는 경우)
export NGD_DATA_ROOT=/path/to/ngd_data     # Windows: set NGD_DATA_ROOT=C:\ngd_data

python paths.py                        # 경로 설정 확인
python analysis/step1_conf_explorer.py
python analysis/step2a_channel_selection.py
python analysis/step2b_signal_separation.py
python analysis/step3_threshold_analysis.py
python analysis/step4_ratio_threshold.py
python analysis/roc_analysis.py
python analysis/negneg_false_positive_check.py
```

`NGD_DATA_ROOT` 로 지정하는 것은 데이터 폴더의 **상위 경로**이며, 그 아래에
`dataset/` 이 있어야 한다. 즉 `NGD_DATA_ROOT=/path/to/ngd_data` 라면
`/path/to/ngd_data/dataset/solo/train/neg` 와 같은 구조다.

`python paths.py`를 실행하면 각 데이터 폴더의 이미지 개수와 모델 파일이
제대로 잡히는지 확인할 수 있다.

촬영 조건 최적화 실험은 별도로 실행한다. 본 파이프라인과 무관하며,
`dataset/` 이 아니라 `raw_pilot/` 의 촬영 원본을 읽는다.

```bash
python analysis/pilot/pilot_1.py    # 조명 파장
python analysis/pilot/pilot_2.py    # 광학 배율
python analysis/pilot/pilot_3.py    # 회전 각도
python analysis/pilot/pilot_4.py    # 조명 강도
python analysis/pilot/pilot_6.py    # 이중 시료 배치
streamlit run analysis/pilot/pilot_5.py   # 기기 비교 결과 뷰어
```

pilot 산출물은 프리뷰 이미지가 많아 137 MB 에 달하므로 저장소가 아니라
`$NGD_DATA_ROOT/pilot_outputs/` 아래에 저장된다.

검출 오버레이 이미지는 기본적으로 저장하지 않는다. 필요하면 `--save_viz`를
붙인다. 다만 이미지 수만큼 파일이 생성된다.

---

## 모델

- 아키텍처: **YOLOv8s** (YOLOv8n이 아니다 — 논문 기재 오류)
- 클래스: `roi`, `tube` — **2개** (논문 기재 3개와 다르다)
- 학습: 300 epoch 설정, patience 90으로 **213 epoch에서 조기 종료** (best는 123 epoch)
- 입력 크기: 640, seed 0
- ultralytics 8.3.171, 학습 완료 2025-11-04 03:13 KST
  (가중치 메타데이터에는 2025-11-03T18:13 UTC 로 기록되어 있다)

제출 당시 폴더에는 `weights.pt`(초기 1클래스 모델)와 `new_weights.pt`(2클래스)가
함께 있었고 실제로 사용한 것은 후자다. 이 저장소의 `models/weights.pt` 는
`new_weights.pt` 를 옮기며 이름을 바꾼 파일이다.

### 학습 데이터셋

Roboflow 로 구축한 학습 데이터셋 **v8** (2025-11-04, 700장).

> v2 / v8 / v10 은 이 프로젝트에서 데이터셋 세대를 구분하기 위해 쓰는
> 자체 버전 번호다. Roboflow 프로젝트는 비공개이며, 공개된 학습 데이터는
> Zenodo 데이터셋에 포함되어 있다.

- 원본 300개를 train 200 / valid 50 / test 50 으로 분할
- train 에만 증강 적용 (회전 ±15°, 밝기 ±15%) 3배 → 600장
- valid 50장, test 50장은 원본 그대로 (600 + 50 + 50 = 700)

분할을 직접 확인한 결과 **동일 원본이 여러 split 에 들어간 사례는 없었다.**
파일명 앞부분이 겹치는 경우가 있으나 서로 다른 이미지다. 원본 수는 증강을
적용하지 않은 별도 버전(v10)을 생성해 300장임을 확인했다.

Roboflow 보고 성능은 mAP@50 99.9%, Precision 99.1%, Recall 100%, F1 99.6% 다.
논문에 기재된 mAP@50 99.0% 는 실제 값과 다르다.

검출 모델은 양성/음성을 판별하지 않는다. 튜브와 ROI의 위치만 찾고,
판정은 전적으로 ROI의 형광값 비율로 이루어진다.

---

## 안전 장치

진단 도구에서는 잘못된 판정보다 판정 불가가 안전하다.
아래 상황에서는 결과를 내지 않고 재촬영을 안내한다.

| 상황 | 판단 근거 |
|---|---|
| ROI가 두 개 검출되지 않음 | 용액이 흩어졌거나 튜브가 하나만 담김 |
| 상단 밝기가 221.0 초과 | 위쪽이 NC가 아닐 수 있음 (튜브 순서 반전) |
| 어느 한쪽 ROI가 포화 | 8비트 상한(255)을 넘어 실제 형광 세기를 알 수 없음 |
| 튜브가 3개 이상 검출 | 사용자가 의도한 조합인지 확인 필요 |

판정이 나오더라도 ratio가 임계값에서 0.05 이내면 경계값으로 보고 재촬영을 권한다.
음성 시료끼리 촬영해도 ratio가 ±0.089 정도 흔들리므로, 이 범위의 결과는
재촬영 시 뒤집힐 수 있다.

---

## 한계

학부생 팀이 약 3개월 동안 수행한 프로젝트이며, 아래는 이후 코드와 결과를
다시 검토하며 확인한 한계다.

### 검증 범위

**검출 한계(LOD)를 확립하지 못했다.** 농도 의존적 형광 패턴이 명확히
나타나지 않아, 특정 농도에서 검출되었다는 단일 관찰만 있다.

**특이도 검증에 인간 gDNA만 사용했다.** 표적으로 삼은 `porA` pseudogene은
본래 임질균을 다른 *Neisseria* 종과 구별하기 위한 것이므로, 근연종
(*N. meningitidis* 등)에 대한 교차 반응 확인이 필요하나 수행하지 못했다.

**실제 임상 검체가 아닌 합성 표적 DNA로 검증했다.** 진단 도구가 아니라
개념 검증(proof-of-concept) 수준이다.

**촬영 조건 최적화 실험은 조건당 2~20장으로 수행했으나** 대부분의 조건에서
통계적 유의성을 확보하지 못했다. 조명 강도는 세 조건 모두 p > 0.05 였고,
배율은 2x 에서만 p < 0.05 였다(p = 0.0077). 다만 2.5x 가 p = 0.0503 으로
경계에 거의 붙어 있어, 표본이 늘면 이 구분은 뒤집힐 수 있다. 자세한 내용은
[analysis/pilot/README.md](analysis/pilot/README.md) 참고.

### 통계적 한계

**절대 기준선이 단일 극단값에 의존한다.** cutoff 221.0은 train 음성 40장의
99.7 백분위수인데, n=40에서 이 값은 최댓값과 같아진다. 실제로 train 음성의
최댓값이 정확히 221.0이다. 다만 양성 정보까지 활용하는 Youden's J로
재산출한 값이 221.5로 거의 같아, 방법 자체는 타당한 것으로 확인됐다.

**비율 임계값의 신뢰구간이 넓다.** 부트스트랩 95% CI가 [1.09, 1.16]으로,
이 범위 안에서는 어느 값을 택하든 성능 차이가 통계적으로 유의하지 않다.

**모든 정확도 수치의 신뢰구간이 넓다.** 위 성능 표를 참고하면 된다.

**test set 양성 15장 중 5장이 포화(G_p95 = 255)됐다.** 포화된 값은 무조건
기준선을 넘으므로 사실상 자동 정답이 된다. 기본 조건 이미지만으로 계산하면
정확도는 96.7%가 아니라 95.0%다.

### 시스템 견고성

**판정 마진이 측정 변동과 비슷한 크기다.** 양성 판정 마진은 약 11.6%인데,
음성 시료끼리 촬영해도 ratio가 ±8.9% 흔들린다. 위양성 4건이 모두 여기서
발생했으며, 임계값 조정으로는 해결되지 않는다.

**검출 모델에 기기 편향이 있다.** Galaxy Note 8로 촬영한 이미지의
두 번째 ROI confidence가 0.656~0.793인 반면 iPhone 계열은 0.797~0.815다.
이 때문에 정상 이미지 3장이 검출 임계값(0.70)에 미달해 판정에서 제외됐다.
학습 데이터에 해당 기기 이미지를 보강해 재학습하는 것이 근본 해결이다.

**위쪽 튜브 QC 가 절대 기준 하나에 의존한다.** 검증 데이터에 위쪽이 양성인
이미지가 없어 이 경로의 정확도를 확인할 수 없다. 실제로 위쪽에 양성을 놓은
이미지([data/samples/limitation_upper_pos_lower_half.jpg](data/samples/))에서
상단 밝기가 217로 경고 기준(221.0)에 4만큼 미달해 안전장치가 작동하지 않았고,
판정이 그대로 출력됐다. 정상 음성 쌍의 상단이 202~203이므로 기준을 낮추면
정상 이미지까지 걸린다.

### 촬영 품질 검증

기기 호환성 검증 이미지 22장 중 9장이 ROI 검출에 실패했다. 그중 4장은
의도적으로 촬영한 오류 이미지지만, 나머지 5장은 정상 이미지다.

한편 촬영 품질 검증용 이미지(`qc_test`)는 대부분 단일 튜브로 촬영되어,
pair를 전제하는 현재 판정 방식에서는 품질과 무관하게 판정 불가로 처리된다.
따라서 이 데이터로는 QC 검출 성능을 평가할 수 없다.
특히 흐린 pair 이미지가 하나도 없어, 초점이 맞지 않은 이미지가 검출을 통과해
잘못된 판정을 내는지는 확인되지 않았다.

---

## 문서

- [CHANGELOG.md](CHANGELOG.md) — 논문 제출 이후 변경 사항과 근거
- [data/README.md](data/README.md) — 데이터셋 구조, 촬영 조건, 라벨 규칙
- [data/samples/README.md](data/samples/README.md) — 예시 이미지 9장의 기대 결과
- [analysis/pilot/README.md](analysis/pilot/README.md) — 촬영 조건 최적화 실험
- [results/README.md](results/README.md) — 분석 결과 폴더 구조