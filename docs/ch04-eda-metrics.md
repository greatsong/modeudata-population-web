# 4장. 탐색적 분석과 지표 만들기

← [3장. 데이터 불러오고 정제하기](ch03-load-clean.md) | 다음 → [5장. 시각화: 인구 피라미드](ch05-visualize.md)

---

## 🎯 학습 목표
- 데이터를 "그냥 보는" 탐색적 분석(EDA)을 한다.
- 원시 인구 숫자에서 **의미 있는 지표**를 계산한다: 총인구·평균연령·고령화율·중위연령·소멸위험지수.
- numpy로 여러 동네의 지표를 **한 번에** 계산한다.

---

## 0. 준비: 인구 행렬 만들기

3장에서 정제한 `df`, `male_cols`, `female_cols`, `ages` 가 있다고 합시다.
인구 열들을 numpy 배열로 꺼내면 계산이 쉬워집니다.

```python
import numpy as np

M = df[male_cols].to_numpy(float)     # (동네수, 101)  남자
F = df[female_cols].to_numpy(float)   # (동네수, 101)  여자
T = M + F                             # (동네수, 101)  남+여 합
ages_arr = np.array(ages)             # [0,1,...,100]
```

이제 `T[i]` 는 i번째 동네의 "나이별 인구 곡선"입니다.

## 1. 총인구

```python
total = T.sum(axis=1)        # 동네별 총인구
df["총인구"] = total
df[["행정구역", "총인구"]].sort_values("총인구", ascending=False).head()
```

## 2. 평균연령

평균연령 = (각 나이 × 그 나이 인구)의 합 ÷ 총인구. 가중평균이에요.

```python
mean_age = (T * ages_arr).sum(axis=1) / np.clip(total, 1, None)
df["평균연령"] = mean_age
```

> `np.clip(total, 1, None)` 은 "0으로 나누기"를 막는 안전장치입니다(최솟값을 1로).

## 3. 고령화율과 유소년 비율

```python
old   = T[:, 65:].sum(axis=1) / np.clip(total, 1, None) * 100   # 65세 이상 비율(%)
youth = T[:, :15].sum(axis=1) / np.clip(total, 1, None) * 100   # 0~14세 비율(%)
df["고령화율"] = old
df["유소년비율"] = youth
```

## 4. 중위연령

인구를 한 줄로 세웠을 때 **딱 가운데 사람의 나이**입니다. 누적합으로 구해요.

```python
def median_age(row_counts):
    half = row_counts.sum() / 2
    acc = 0
    for a in range(101):
        acc += row_counts[a]
        if acc >= half:
            return a
    return 100

df["중위연령"] = [median_age(T[i]) for i in range(len(df))]
```

## 5. ⭐ 소멸위험지수

이 교재의 대표 지표입니다. **"20~39세 여성 ÷ 65세 이상 인구"** 로 정의해요.
젊은 여성(아이를 낳을 핵심 세대)이 노인보다 적으면 그 지역은 인구가 줄어들 위험이 큽니다.

```python
women_20_39 = F[:, 20:40].sum(axis=1)            # 20~39세 여성
elderly     = T[:, 65:].sum(axis=1)              # 65세 이상 전체
ext = women_20_39 / np.clip(elderly, 1, None)
df["소멸위험지수"] = ext
```

이 값을 단계로 나누면 해석이 쉬워집니다.

```python
def risk_band(v):
    if v < 0.2:  return "소멸 고위험"
    if v < 0.5:  return "소멸위험 진입"
    if v < 1.0:  return "소멸 주의"
    if v < 1.5:  return "보통"
    return "저위험"

df["위험단계"] = df["소멸위험지수"].apply(risk_band)
```

> 📌 **지수 < 0.5** 면 학계에서 '소멸위험지역'으로 봅니다.

## 6. 노령화지수·노년부양비 (보너스)

```python
df["노령화지수"] = T[:, 65:].sum(1) / np.clip(T[:, :15].sum(1), 1, None) * 100   # 노인/유소년
df["노년부양비"] = T[:, 65:].sum(1) / np.clip(T[:, 15:65].sum(1), 1, None) * 100  # 노인/생산연령
```

## 7. 탐색해 보기: 어떤 동네가 가장 늙었나

```python
leaf = df[df["행정구역"].str.endswith(("동", "읍", "면"))]   # 읍·면·동만
print("가장 늙은 동네 TOP5")
print(leaf.nlargest(5, "평균연령")[["행정구역", "평균연령", "소멸위험지수"]])

print("\n가장 젊은 동네 TOP5")
print(leaf.nsmallest(5, "평균연령")[["행정구역", "평균연령", "소멸위험지수"]])
```

> 이렇게 데이터를 이리저리 들여다보는 게 **EDA(탐색적 데이터 분석)** 입니다.
> 모델을 돌리기 전에 데이터와 친해지는 단계예요.

---

## 🌐 웹앱에서는?

웹앱의 `stats(row)` 함수가 위 지표를 **한 동네에 대해** 똑같이 계산합니다(`index.html`).
`band(idx)` 함수는 우리의 `risk_band` 와 글자 하나까지 같습니다.
화면 위쪽 카드(총인구·평균연령·고령화율…)와 반원 게이지가 바로 이 값들이에요.

## 🧪 직접 해보기
1. 우리 동네의 모든 지표를 한 줄로 출력해 보세요: `df[df["행정구역"].str.contains("우리동네이름")]`
2. 소멸위험지수가 0.2 미만인 '소멸 고위험' 읍·면·동은 전국에 몇 개인가요?
3. 평균연령과 소멸위험지수는 어떤 관계일까요? (다음 장에서 그래프로 확인합니다.)

---
다음 → [5장. 시각화: 인구 피라미드](ch05-visualize.md)
