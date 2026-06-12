# 5장. 시각화: 인구 피라미드

← [4장. 탐색적 분석과 지표 만들기](ch04-eda-metrics.md) | 다음 → [6장. 닮은 동네 찾기 (유사도)](ch06-similarity.md)

---

## 🎯 학습 목표
- matplotlib로 그래프를 그린다.
- **인구 피라미드**(나이·성별 막대그래프)를 그린다.
- 연령 분포 곡선, 두 동네 비교, 산점도를 그린다.
- "좋은 그래프"의 조건을 안다.

---

## 0. 한글 폰트 설정 (안 그러면 글자가 깨져요)

```python
import matplotlib.pyplot as plt
plt.rcParams["axes.unicode_minus"] = False
# 윈도우: plt.rcParams["font.family"] = "Malgun Gothic"
# 맥:     plt.rcParams["font.family"] = "AppleGothic"
# 코랩/리눅스: !apt-get install -y fonts-nanum 후 "NanumGothic"
```

## 1. 한 동네 고르기

```python
name = "서울특별시 강남구 역삼1동"   # 원하는 동네로 바꾸세요
i = df.index[df["행정구역"] == name][0]
male   = M[i]    # 길이 101 (남자, 나이별)
female = F[i]    # 길이 101 (여자, 나이별)
```

## 2. 인구 피라미드 그리기

피라미드의 핵심 아이디어: **남자는 음수(왼쪽), 여자는 양수(오른쪽)** 로 가로 막대를 그린다.

```python
fig, ax = plt.subplots(figsize=(7, 9))
ax.barh(ages, -male,   color="#4C8DF6", label="남자")   # 왼쪽
ax.barh(ages,  female, color="#F4789B", label="여자")   # 오른쪽
ax.set_xlabel("인구 수");  ax.set_ylabel("나이(세)")
ax.set_title(f"{name} 인구 피라미드")

# x축 음수 라벨을 양수로 보이게
xt = ax.get_xticks()
ax.set_xticklabels([f"{abs(int(x)):,}" for x in xt])
ax.legend();  plt.tight_layout();  plt.show()
```

### 읽는 법
- **위가 넓다** → 노인이 많은 늙은 동네
- **아래가 넓다** → 아이·청년이 많은 젊은 동네
- **특정 나이에 불쑥 튀어나옴** → 대학가(20대), 신도시(30대+0~9세) 등 특징

## 3. 연령 분포 곡선

남녀를 합친 나이별 인구를 선으로 그리면 동네의 "나이 지문"이 됩니다.

```python
total_by_age = male + female
fig, ax = plt.subplots(figsize=(9, 4))
ax.fill_between(ages, total_by_age, color="#F4B400", alpha=0.3)
ax.plot(ages, total_by_age, color="#E8930C")
peak = int(total_by_age.argmax())
ax.annotate(f"최다 {peak}세", (peak, total_by_age[peak]),
            textcoords="offset points", xytext=(0, 8), ha="center")
ax.set_xlabel("나이(세)"); ax.set_ylabel("인구 수"); ax.set_title(f"{name} 연령 분포")
plt.tight_layout(); plt.show()
```

## 4. 두 동네 비교 (비율로)

인구 규모가 다른 두 동네는 **비율(%)** 로 맞춰 비교해야 공정합니다.

```python
def age_ratio(idx):
    t = T[idx]
    return t / t.sum() * 100      # 합이 100%가 되도록

a = df.index[df["행정구역"] == "서울특별시 강남구 역삼1동"][0]
b = df.index[df["행정구역"] == "부산광역시 해운대구 좌동"][0]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(ages, age_ratio(a), label=df.loc[a, "행정구역"], color="#E8930C")
ax.plot(ages, age_ratio(b), label=df.loc[b, "행정구역"], color="#4C8DF6")
ax.set_xlabel("나이(세)"); ax.set_ylabel("비율(%)"); ax.legend()
plt.tight_layout(); plt.show()
```

> 💡 4장에서 던진 질문 — "평균연령과 소멸위험지수의 관계?" — 을 **산점도**로 확인해 봅시다.

```python
leaf = df[df["행정구역"].str.endswith(("동","읍","면"))]
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(leaf["평균연령"], leaf["소멸위험지수"], s=6, alpha=0.3)
ax.axhline(0.5, color="red", ls="--", label="소멸위험 기준 0.5")
ax.set_xlabel("평균연령"); ax.set_ylabel("소멸위험지수"); ax.legend()
plt.tight_layout(); plt.show()
```

평균연령이 높을수록 소멸위험지수가 낮아지는 **뚜렷한 음의 관계**가 보일 거예요.

## 5. 좋은 그래프의 조건
- **제목과 축 이름**이 있다 (무슨 그래프인지 5초 안에 안다)
- **단위**가 분명하다 (명? %?)
- **비교 대상은 같은 기준**으로 (규모 다르면 비율로)
- 색이 의미를 돕는다 (남=파랑, 여=분홍처럼 직관적으로)

---

## 🌐 웹앱에서는?

| 우리가 그린 것 | 웹앱 함수 | 라이브러리 |
|---|---|---|
| 인구 피라미드 | `pyramid(row)` | Plotly (`barmode:'relative'`) |
| 연령 분포 | `distribution(row)` | Plotly |
| 두 동네 비교 | `compare()` | Plotly |

원리(남자 음수·여자 양수, 비율 비교)는 라이브러리가 달라도 똑같습니다.
matplotlib로 원리를 익히면 어떤 도구로도 그릴 수 있어요.

## 🧪 직접 해보기
1. 우리 동네와 옆 동네의 피라미드를 나란히 그려 비교해 보세요.
2. 가장 젊은 동네와 가장 늙은 동네의 분포 곡선을 한 그래프에 겹쳐 보세요.
3. 산점도에서 "평균연령은 높은데 소멸위험지수도 높은" 예외 동네가 있나요? 왜 그럴까요?

---
다음 → [6장. 닮은 동네 찾기 (유사도)](ch06-similarity.md)
