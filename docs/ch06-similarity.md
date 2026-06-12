# 6장. 닮은 동네 찾기 (유사도)

← [5장. 시각화: 인구 피라미드](ch05-visualize.md) | 다음 → [7장. 머신러닝 준비: 특징과 스케일링](ch07-features-scaling.md)

---

## 🎯 학습 목표
- 동네를 **벡터(숫자 목록)** 로 본다 — 머신러닝으로 가는 결정적 한 걸음.
- **코사인 유사도**로 "모양이 닮은 정도"를 잰다.
- 선택한 동네의 '쌍둥이 동네'를 찾는다.

> 이 장은 비지도학습(8장)으로 가는 다리입니다. "동네 = 숫자 벡터"라는 관점을 여기서 확실히 잡으세요.

---

## 1. 동네를 "벡터"로 보기

한 동네의 나이별 인구 비율은 길이 101짜리 숫자 목록입니다.

```
역삼1동 = [0세 0.9%, 1세 0.8%, ..., 100세 0.01%]   ← 101개 숫자
```

이 101개 숫자를 **101차원 공간의 한 점(벡터)** 으로 생각할 수 있어요.
점이 가까우면 인구 구조가 닮은 것이고, 멀면 다른 거죠. 이게 머신러닝의 핵심 직관입니다.

```python
import numpy as np
def ratio_vec(idx):
    t = T[idx].astype(float)
    return t / t.sum()        # 길이 101, 합이 1
```

## 2. "닮음"을 어떻게 숫자로 잴까: 코사인 유사도

두 벡터가 이루는 **각도**로 닮음을 잽니다.
- 같은 방향(각도 0°) → 코사인 = **1** (완전히 닮음)
- 직각(90°) → 코사인 = **0** (전혀 다름)

규모(크기)는 무시하고 **모양(방향)** 만 보기 때문에, 인구가 많은 동네와 적은 동네도 공정하게 비교됩니다.

```
코사인 유사도 = (A·B) / (|A| × |B|)
```

```python
def cosine(a, b):
    return float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
```

> `a @ b` 는 내적(elementwise 곱의 합), `np.linalg.norm` 은 벡터 길이입니다.

## 3. 쌍둥이 동네 찾기

읍·면·동만 모아 놓고, 선택한 동네와 가장 닮은 5곳을 찾습니다.

```python
leaf = df[df["행정구역"].str.endswith(("동","읍","면"))].reset_index(drop=True)
Tl = (leaf[male_cols].to_numpy(float) + leaf[female_cols].to_numpy(float))
Rl = Tl / Tl.sum(axis=1, keepdims=True)     # (동네수, 101) 비율 벡터

target = "서울특별시 강남구 역삼1동"
ti = leaf.index[leaf["행정구역"] == target][0]
tv = Rl[ti]

# 모든 동네와의 코사인 유사도를 한 번에
norms = np.linalg.norm(Rl, axis=1)
sims = (Rl @ tv) / (norms * np.linalg.norm(tv) + 1e-12)

leaf["유사도"] = sims
twins = leaf.drop(index=ti).nlargest(5, "유사도")
print(twins[["행정구역", "유사도", "평균연령", "소멸위험지수"]])
```

선택한 동네와 닮은꼴 5곳이 유사도 순으로 나옵니다. 보통 0.99 이상이면 거의 쌍둥이예요.

## 4. 겹쳐 그려 확인

진짜 닮았는지 분포 곡선을 겹쳐 봅시다.

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(ages, Rl[ti]*100, lw=3, color="#E8930C", label="⭐ "+target)
for j in twins.index:
    ax.plot(ages, Rl[j]*100, alpha=0.7, label=leaf.loc[j, "행정구역"])
ax.set_xlabel("나이(세)"); ax.set_ylabel("비율(%)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
```

선들이 거의 포개지면 "인구 구조 쌍둥이"가 맞습니다.

## 5. 유사도 → 군집화로

쌍둥이 찾기는 "한 동네와 닮은 이웃"을 찾는 일이었습니다.
그런데 만약 **전체를 비슷한 무리로 한꺼번에 나누고** 싶다면? 그게 바로 **군집화(8장)** 입니다.
유사도가 1:1 비교라면, 군집화는 전체를 그룹으로 묶는 일이에요. 같은 직관(벡터·거리)에서 출발합니다.

---

## 🌐 웹앱에서는?

웹앱의 **👯 쌍둥이 동네** 탭이 이 장 그대로입니다(`twins(row)` 함수).
- 각 동네를 비율 벡터로 만들고 **단위벡터로 정규화**(`unit`)해 둔 뒤
- 내적으로 코사인 유사도를 계산해 상위 5곳을 표와 겹친 그래프로 보여 주고,
- 🗺️ 지도에 쌍둥이들의 실제 위치까지 찍어 줍니다(`twinMap`).
"멀리 떨어져 있어도 인구 구조는 닮았다"는 걸 눈으로 보여 주는 탭이에요.

## 🧪 직접 해보기
1. 우리 동네의 쌍둥이 5곳은 어디인가요? 지리적으로 가까운가요, 먼가요?
2. 코사인 유사도 대신 **유클리드 거리**(`np.linalg.norm(Rl - tv, axis=1)`)로 가장 가까운 5곳을 찾아 비교해 보세요. 결과가 다른가요?
3. 왜 유사도를 잴 때 인구 '수'가 아니라 '비율'을 썼을까요?

---
다음 → [7장. 머신러닝 준비: 특징과 스케일링](ch07-features-scaling.md)
