# 9장. 차원 축소와 시각화 (PCA)

← [8장. 군집화: K-평균 알고리즘](ch08-kmeans.md) | 다음 → [10장. 군집 해석하고 지도에 그리기](ch10-interpret-map.md)

---

## 🎯 학습 목표
- "차원이 높다"는 게 왜 문제인지 안다.
- **주성분 분석(PCA)** 으로 202차원을 2차원으로 줄인다.
- 군집을 2차원 산점도로 그려 눈으로 확인한다.

---

## 1. 문제: 202차원을 어떻게 보지?

8장에서 동네를 202차원(또는 203차원) 점으로 묶었습니다. 그런데 사람은 2~3차원만 볼 수 있어요.
202개 축을 종이에 그릴 순 없죠. 그래서 **차원을 줄여야** 합니다. 단, 정보는 최대한 지키면서요.

## 2. PCA의 아이디어: "가장 잘 보이는 각도에서 사진 찍기"

3차원 물체(예: 주전자)를 2차원 사진으로 찍는다고 합시다.
- 위에서 찍으면 동그라미만 보여 정보 손실이 큽니다.
- 옆에서 찍으면 손잡이·주둥이까지 보여 잘 구별됩니다.

**PCA**는 데이터가 **가장 넓게 퍼져 보이는 방향(=정보가 가장 많은 방향)** 을 찾아 그 방향으로 사진을 찍습니다.
- **제1주성분(PC1)**: 데이터가 가장 많이 퍼진 방향
- **제2주성분(PC2)**: PC1과 직각이면서 그다음으로 많이 퍼진 방향

이 두 축에 점을 투영하면, 202차원의 "흩어진 모양"을 최대한 보존한 2차원 그림이 됩니다.

## 3. 코드: 두 줄로 차원 축소

7장의 표준화된 특징을 그대로 씁니다(**PCA 전 표준화는 필수**).

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=42)
P = pca.fit_transform(X_sexage_s)        # (N, 2)  각 동네의 (PC1, PC2)

print("설명된 분산 비율:", pca.explained_variance_ratio_.round(3))
print("두 축이 담은 정보:", pca.explained_variance_ratio_.sum().round(3))
```

`explained_variance_ratio_` 는 "각 축이 원래 정보의 몇 %를 담았는가"입니다.
두 축 합이 0.5라면 202차원의 정보 절반을 2차원에 담았다는 뜻이에요(나머지는 평면 뒤로 숨음).

## 4. 군집을 산점도로 그리기

8장의 `labels` 와 합쳐, 군집을 색으로 칠한 2차원 지도를 그립니다.

```python
import matplotlib.pyplot as plt
colors = ["#E8930C","#4C8DF6","#54A24B","#E45756","#B279A2","#3BA6A0","#F58518","#9C755F"]

fig, ax = plt.subplots(figsize=(7, 6))
for c in range(K):
    m = labels == c
    ax.scatter(P[m, 0], P[m, 1], s=10, alpha=0.6,
               color=colors[c % len(colors)], label=f"군집 {c+1} ({m.sum()})")
ax.set_xlabel("제1주성분 (PC1)"); ax.set_ylabel("제2주성분 (PC2)")
ax.set_title("동네 군집 (PCA 2차원)"); ax.legend()
plt.tight_layout(); plt.show()
```

같은 색 점들이 한쪽에 모여 있고 색끼리 잘 갈라져 있으면, 군집이 잘 나뉜 것입니다. 🎉

## 5. 군집별 평균 분포로 "성격" 확인

산점도가 추상적이라면, 각 군집의 **평균 연령 곡선**이 더 직관적입니다.

```python
fig, ax = plt.subplots(figsize=(9, 4))
for c in range(K):
    mean_curve = (Tl[labels == c] / Tl[labels == c].sum(1, keepdims=True)).mean(0) * 100
    ax.plot(ages, mean_curve, color=colors[c % len(colors)], lw=2, label=f"군집 {c+1}")
ax.set_xlabel("나이(세)"); ax.set_ylabel("평균 연령 비율(%)"); ax.legend()
plt.tight_layout(); plt.show()
```

어떤 군집은 20대에 봉우리(대학가), 어떤 군집은 노년에 봉우리(고령 시골)… 곡선 모양이 곧 군집의 정체입니다.

## 6. PCA, 이것만 기억하세요
- **목적**: 고차원을 사람이 볼 수 있게 2~3차원으로 줄이기 (시각화·압축)
- **원리**: 정보(분산)가 가장 많은 방향부터 새 축으로 삼기
- **주의**: 반드시 **표준화 후** 적용. 새 축(PC1·PC2)은 원래 나이 하나가 아니라 여러 특징의 혼합이라 해석은 조심.

---

## 🌐 웹앱에서는?

웹앱엔 scikit-learn이 없어 PCA도 **직접 구현**했습니다(`pca2(X)` 함수).
- **거듭제곱 반복(power iteration)** 이라는 기법으로 제1·제2 주성분을 찾고 그 축에 점을 투영합니다.
- 결과를 `cluScatterPlot` 이 군집 색으로 산점도(🧩 군집 분석 탭)로 그립니다.
scikit-learn은 더 정교한 방법(SVD)을 쓰지만, "정보가 많은 방향을 찾아 투영"하는 원리는 똑같습니다.

## 🧪 직접 해보기
1. `X_age_s`(101)와 `X_sexage_size_s`(203)의 `explained_variance_ratio_` 를 비교해 보세요. 어느 쪽이 2차원에 더 잘 담기나요?
2. 우리 동네 점을 산점도에서 별표(★)로 크게 표시해 보세요. (`ax.scatter(P[ti,0], P[ti,1], marker="*", s=300)`)
3. `n_components=3` 으로 3차원 PCA를 해서 3D 산점도로 그려 보세요(`projection="3d"`).

---
다음 → [10장. 군집 해석하고 지도에 그리기](ch10-interpret-map.md)
