# 분석 결과

`analysis/` 의 스크립트를 실행하면 생성되는 결과물입니다.
수치 결과(csv, json)와 그림(png)만 저장소에 포함하고,
중간 산출 이미지(검출 오버레이 등)는 용량 문제로 제외했습니다.

---

## 폴더 구조

```
results/
├── step1_conf/
│   └── per_image_summary.csv        이미지별 검출 confidence
│
├── step2a_channel/
│   ├── channel_scan_values.csv      14개 조합의 ROI 값
│   ├── channel_scan_report.csv      조합별 Cohen's d
│   └── best_channel.json            선정 결과
│
├── step2b_separation/
│   ├── Gp95_values.csv              pos/neg 별 G_p95
│   └── Gp95_stats.csv               Welch t-test 결과
│
├── step3_threshold/
│   ├── solo_values.csv
│   ├── neg_baseline_stats.json      정규성 검정 + cutoff
│   └── neg_baseline_qq.png
│
├── step4_ratio/
│   └── (pair 분석 결과)
│
├── roc/
│   ├── solo_roc_curve.csv / .png
│   └── pair_roc_curve.csv / .png
│
└── negneg_check/
    └── neg_neg_pair_analysis.csv    음성-음성 쌍 ratio
```

---

## 주요 결과값

| 항목 | 값 | 출처 |
|---|---|---|
| 검출 confidence 임계 | 0.70 | step1 |
| 형광 지표 | G채널 p95 | step2a |
| 음성 절댓값 cutoff | 221.0 a.u. | step3 |
| 양성 판정 비율 임계 | 1.148 | step4 |

---

## 읽을 때 유의할 점

- 표본 수가 작습니다. 정확도 등의 수치에는 신뢰구간을 함께 보아야 하며,
  현재 결과 파일에는 구간이 포함되어 있지 않습니다.
- `step3` 의 cutoff 는 train 음성 데이터의 99.7 백분위수로 산출했습니다.
  표본이 적을 때 이 방식은 극단값 하나에 크게 좌우됩니다.
- `roc/` 의 AUC 는 임계값 선택과 무관한 지표이므로, 임계값에 의존하는
  정확도보다 성능을 보수적으로 보여줍니다.
