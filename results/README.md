# 분석 결과

`analysis/` 의 스크립트를 실행하면 생성되는 결과물입니다.
수치 결과(csv, json)와 그림(png)만 저장소에 포함하고,
중간 산출 이미지(검출 오버레이 등)는 용량 문제로 제외했습니다.

---

## 폴더 구조

```
results/
├── step1_conf/                      검출 confidence 스윕
│   ├── per_image_summary.csv          이미지별 검출 confidence
│   ├── threshold_sweep.csv            conf 0.50–0.85 스윕, 그룹별 검출 수
│   └── borderline_images.txt          임계값 부근에 걸린 이미지와 conf 값
│
├── step2a_channel/                  형광 지표 선정
│   ├── channel_scan_values.csv        14개 조합(7채널 × 2지표)의 ROI 값
│   ├── channel_scan_report.csv        조합별 Cohen's d
│   └── best_channel.json              선정 결과 (G채널 p95)
│
├── step2b_separation/               양성/음성 분리 검정
│   ├── Gp95_values.csv                pos/neg 별 G_p95
│   ├── Gp95_stats.csv                 Welch t-test 결과
│   └── summary.json                   요약
│
├── step3_threshold/                 절대 음성 기준선
│   ├── solo_values.csv                solo 전체 G_p95
│   ├── neg_train_values.csv           기준선 산출에 쓴 train 음성 40장
│   ├── neg_baseline_stats.json        정규성 검정 + cutoff
│   ├── neg_baseline_qq.png            정규성 Q-Q plot
│   └── summary.json                   요약
│
├── step4_ratio/                     비율 판정 임계값
│   ├── pair_analysis.csv              pair 44장의 Iu / Il / ratio
│   ├── solo_analysis.csv              solo 대조용 값
│   ├── threshold_derivation.json      임계값 도출 근거 (A1 / A2 / Youden)
│   ├── test_all_eval.csv              기기 호환성 검증 세트 평가
│   └── summary.json                   요약
│
├── roc/                             ROC / AUC (임계값 비의존 지표)
│   ├── solo_roc_train.csv / .png      solo train
│   ├── solo_roc_test.csv / .png       solo test
│   ├── solo_roc_all.csv / .png        solo 전체
│   ├── pair_roc.csv / .png            pair (ratio 기준)
│   ├── roc_values.csv                 ROC 산출에 쓴 이미지별 label / score
│   └── summary.json                   AUC + 부트스트랩 신뢰구간
│
└── negneg_check/                    음성-음성 쌍 위양성 확인
    ├── negneg_analysis.csv            음성-음성 쌍 24장의 ratio
    ├── false_positives.csv            위양성으로 판정된 4장
    └── summary.json                   위양성률 + 신뢰구간
```

`pair_analysis.csv` 의 `note` 열에는 분석에서 제외된 이미지의 사유가
`ROI_PARTIAL` 등으로 기록되어 있습니다. 음성-음성 쌍 26장 중 2장이
여기에 해당해 기준 분포는 24장입니다.

---

## 주요 결과값

| 항목 | 값 | 출처 |
|---|---|---|
| 검출 confidence 임계 | 0.70 | step1 |
| 형광 지표 | G채널 p95 | step2a |
| 음성 절댓값 cutoff | 221.0 a.u. | step3 |
| **양성 판정 비율 임계** | **1.1162** | step4 |

비율 임계값은 `221.0 / median(pair NC, n=44) = 221.0 / 198.0` 으로 산출한
값입니다. 논문 제출본의 **1.148** 은 `neg_pos` 20장만으로 구한 값이며,
제출 이후 추가 촬영한 `neg_neg` 쌍의 NC 를 기준 분포에 포함해 재계산했습니다.
도출 과정과 변경 근거는 `step4_ratio/threshold_derivation.json` 과
[CHANGELOG.md](../CHANGELOG.md) 를 참고하십시오.

---

## 읽을 때 유의할 점

- 표본 수가 작습니다. 정확도 등의 점추정값만 보면 실제 성능을 과대평가하게
  됩니다. 각 폴더의 `summary.json` 에 부트스트랩 또는 Wilson 신뢰구간이
  함께 들어 있으므로 반드시 같이 보아야 합니다.
- `step3` 의 cutoff 는 train 음성 40장의 99.7 백분위수로 산출했습니다.
  n=40 에서 이 값은 최댓값과 같아지므로, 단일 극단값에 크게 좌우됩니다.
  다만 양성 정보까지 쓰는 Youden's J 로 재산출한 값이 221.5 로 거의 같아
  방법 자체는 타당한 것으로 확인됐습니다.
- `step4_ratio/threshold_derivation.json` 에는 임계값 후보가 세 가지
  (A1: neg_pos 만 / A2: 전체 pair / B: Youden's J) 모두 기록되어 있습니다.
  실제로 채택한 것은 A2 이며, `T_used` 필드에 명시되어 있습니다.
- `roc/` 의 AUC 는 임계값 선택과 무관한 지표이므로, 임계값에 의존하는
  정확도보다 성능을 보수적으로 보여줍니다.
- `negneg_check` 의 위양성률 16.7% 는 Youden's J 로 임계값을 최적화해도
  같은 값이 나옵니다. 임계값 조정으로는 줄일 수 없는, 신호 분포 자체의
  겹침입니다.