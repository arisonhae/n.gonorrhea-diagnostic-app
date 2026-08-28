# 예시 이미지

앱 동작을 확인할 수 있는 이미지 9장이다.
전체 데이터셋 구조는 상위 [data/README.md](../README.md) 참고.

```bash
streamlit run app.py
```

앱을 띄운 뒤 각 이미지를 올리면 아래 결과가 나온다.
모든 값은 실제 측정치이며, 판정 기준은 `RATIO_THR = 1.1162` 이다.

| 파일 | 위 튜브 | 아래 튜브 | Iu | Il | ratio | 판정 |
|---|---|---|---|---|---|---|
| `pair_positive` | NC | 양성 | 164 | 230 | 1.402 | POSITIVE |
| `pair_negative` | NC | 음성 | 203 | 195 | 0.961 | NEGATIVE |
| `pair_borderline_positive` | NC | 양성 | 207 | 240 | 1.159 | POSITIVE + 경계값 경고 |
| `pair_borderline_negative` | NC | 음성 | 202 | 220 | 1.089 | NEGATIVE + 경계값 경고 |
| `pair_false_positive` | NC | 음성 | 169 | 207 | 1.225 | **POSITIVE (오판)** |
| `specificity_human_gdna` | NC | 인간 gDNA | 185 | 167 | 0.903 | NEGATIVE |
| `sensitivity_half_concentration` | NC | 50% 농도 양성 | 169 | 227 | 1.343 | POSITIVE |
| `limitation_upper_pos_lower_half` | **양성** | 50% 농도 양성 | 217 | 214 | 0.986 | NEGATIVE |
| `error_splash` | — | — | — | — | — | 판정 불가 |

---

## 정상 동작

**`pair_positive.jpg`** · **`pair_negative.jpg`**

기본 케이스다. 양성 쌍은 ratio 1.402 로 기준을 여유 있게 넘고,
음성 쌍은 0.961 로 1 근처에 머문다.

**`specificity_human_gdna.jpg`**

아래 튜브에 인간 세포(HEK293T) gDNA 만 넣었다. ratio 0.903 으로 음성이 나온다.
표적이 아닌 DNA 에 반응하지 않는다는 뜻이다.

다만 이 검증은 인간 gDNA 하나로만 이루어졌다. 표적으로 삼은 `porA` pseudogene 은
본래 임질균을 다른 *Neisseria* 종과 구별하기 위한 것이므로, 근연종에 대한
교차 반응 확인이 필요하나 수행하지 못했다.

**`sensitivity_half_concentration.jpg`**

아래 튜브가 표준 농도의 절반이다. ratio 1.343 으로 양성이 나온다.
농도가 낮아도 검출되는 사례이지만, 반복 측정에 근거한 검출 한계(LOD)는
확립하지 못했다.

---

## 경계값

**`pair_borderline_positive.jpg`** → ratio 1.159, 기준에서 **+0.0432**
**`pair_borderline_negative.jpg`** → ratio 1.089, 기준에서 **−0.0271**

둘 다 판정은 나오지만 경계값 경고가 함께 표시된다.

음성 시료끼리 촬영해도 ratio 가 ±0.089 정도 흔들리므로, 기준에서 0.05 이내의
결과는 재촬영하면 뒤집힐 수 있다. 두 이미지는 임계값을 위아래로 감싸고 있어
이 불안정한 구간이 어느 범위인지 보여준다.

---

## 오판

**`pair_false_positive.jpg`** → ratio 1.225, POSITIVE

**위아래 모두 음성인데 양성으로 판정된다.**

상단 NC 의 형광값이 169 로 낮다. 정상 음성 쌍(`pair_negative`)의 상단이 203 인
것과 비교하면 17% 어둡다. 시료가 밝아서가 아니라 **기준 튜브가 어둡게 찍혀서**
비율이 올라간 것이다.

음성-음성 쌍 24장 중 4장에서 이런 일이 발생한다(위양성률 16.7%, 95% CI 6.7–35.9%).
Youden's J 로 임계값을 최적화해도 같은 값이 나오므로, 임계값 조정으로는
없앨 수 없다. 자세한 분석은 [CHANGELOG.md](../../CHANGELOG.md) 참고.

---

## 판정이 성립하지 않는 경우

**`limitation_upper_pos_lower_half.jpg`** → ratio 0.986, NEGATIVE

위 튜브가 양성, 아래가 50% 농도 양성이다. **위쪽에 NC 를 놓지 않은 경우다.**

이 시스템은 `ratio = 아래 / 위` 로 판정하며, **위쪽이 음성 대조군이라는 전제**
위에 서 있다. 위쪽이 이미 양성이면 분모가 커져 비율이 1 아래로 떨어지고,
아래 튜브의 농도와 무관하게 음성으로 나온다.

같은 50% 농도 시료라도 위에 NC 를 놓으면(`sensitivity_half_concentration`)
ratio 1.343 으로 정확히 양성이 나온다.

**문제는 앱이 이를 걸러내지 못한다는 점이다.** 상단 밝기가 217 로,
경고 기준인 221.0 에 4 만큼 미달해 안전장치가 작동하지 않는다.
현재 위쪽 튜브 QC 는 절대 기준 하나에 의존하며, 검증 데이터에 위쪽이 양성인
이미지가 없어 이 경로는 검증되지 않은 상태다.

---

## 판정 불가

**`error_splash.jpg`**

용액이 흩어져 반응 영역(ROI)이 두 개 모두 검출되지 않는다.
앱은 결과를 내지 않고 재촬영을 안내한다.

진단 도구에서는 잘못된 판정보다 판정 불가가 안전하다는 원칙에 따른 것이다.
이 외에도 상단 튜브가 지나치게 밝거나(튜브 순서 반전 의심), 형광이 측정
한계를 넘거나(포화), 튜브가 3개 이상 검출되면 판정을 멈춘다.

---

## 촬영 조건

모든 이미지는 아래 조건으로 촬영했다.

```
iPhone 13 Pro · 2x 배율 · Blue LED 조명 3단계 · transilluminator 표면에서 20 cm · 야간모드 OFF
```

조건을 정한 근거는 [analysis/pilot/README.md](../../analysis/pilot/README.md) 에 있다.