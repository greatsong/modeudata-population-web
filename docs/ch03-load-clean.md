# 3장. 데이터 불러오고 정제하기

← [2장. 데이터 수집하기](ch02-collect.md) | 다음 → [4장. 탐색적 분석과 지표 만들기](ch04-eda-metrics.md)

---

## 🎯 학습 목표
- pandas로 CSV를 불러온다.
- 인코딩·천 단위 콤마·중복·이상한 행 같은 **현실의 더러움**을 처리한다.
- 분석하기 좋은 깨끗한 표를 만든다.

> 💬 데이터 과학자는 시간의 **70~80%** 를 이 "정제"에 씁니다. 화려하진 않지만 가장 중요한 단계예요.

---

## 1. 불러오기

```python
import pandas as pd

df = pd.read_csv("population_2026_05.csv")
print(df.shape)        # (3918, 204)
df.head(3)
```

### 1-1. 인코딩 문제 (한글 깨짐)

행정안전부 원본 파일은 종종 **CP949(EUC-KR)** 인코딩이라 그냥 열면 한글이 깨집니다.
그럴 땐 `encoding` 을 지정하세요.

```python
# 한글이 깨지거나 UnicodeDecodeError가 나면:
df = pd.read_csv("population_2026_05.csv", encoding="cp949")
# 그래도 안 되면 encoding="utf-8-sig" 도 시도
```

> 💡 **BOM(﻿)**: 파일 맨 앞에 보이지 않는 글자가 붙어 첫 열 이름이 `﻿행정구역` 처럼 될 때가 있어요.
> `utf-8-sig` 로 읽거나, 아래처럼 직접 떼어 냅니다.
> ```python
> df.columns = [c.replace("﻿", "").strip() for c in df.columns]
> ```

### 1-2. 천 단위 콤마

원본에 `"1,234"` 처럼 숫자에 콤마가 들어 있으면 글자로 읽힙니다. `thousands` 옵션으로 해결해요.

```python
df = pd.read_csv("population_2026_05.csv", thousands=",")
```

## 2. 열 이름 정리하고 인구 열 모으기

나이·성별 열을 프로그램이 다루기 쉽게 **목록**으로 정의합니다.

```python
df.columns = [c.replace("﻿", "").strip() for c in df.columns]

ages = list(range(101))   # 0,1,...,100
male_cols   = [f"남자_{a}세" for a in range(100)] + ["남자_100세이상"]
female_cols = [f"여자_{a}세" for a in range(100)] + ["여자_100세이상"]

# 모든 열이 실제로 있는지 확인 (없으면 KeyError로 미리 잡기)
missing = [c for c in male_cols + female_cols if c not in df.columns]
print("빠진 열:", missing)        # []  이면 OK
```

> 이렇게 컬럼명을 **규칙으로 생성**하면, 다음 달 파일도 같은 코드로 바로 처리됩니다.

## 3. 더러움 닦기

현실 데이터엔 꼭 이상한 행이 섞여 있습니다. 세 가지를 처리할게요.

### 3-1. 인구를 숫자로 확실히 바꾸기

```python
for c in male_cols + female_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
```

`errors="coerce"` 는 숫자로 못 바꾸는 값을 `NaN`(결측)으로 만들고, `fillna(0)` 으로 0을 채웁니다.

### 3-2. 행정구역 이름 다듬기 (괄호 코드 제거)

원본에선 이름이 `역삼1동(1168064000)` 처럼 코드가 괄호로 붙기도 합니다. 정규식으로 떼어 냅니다.

```python
df["행정구역"] = df["행정구역"].str.replace(r"\s*\(\d+\)\s*$", "", regex=True).str.strip()
```

### 3-3. 중복·이상한 행 제거

세종특별자치시처럼 중복으로 들어오는 행, '출장소' 같은 비표준 행을 정리합니다.

```python
df = df.drop_duplicates(subset="행정구역").reset_index(drop=True)
df = df[~df["행정구역"].str.contains("출장소", na=False)]
print("정제 후 행 수:", len(df))
```

## 4. 위경도 데이터도 불러오기

```python
cent = pd.read_csv("dong_centroids.csv")
cent.columns = [c.replace("﻿", "").strip() for c in cent.columns]   # 코드, lat, lon
cent["코드"] = cent["코드"].astype(str).str.strip()
cent.head(3)
```

나중에 `df["코드"]` 와 이 표의 `코드` 를 `merge` 로 붙여 지도를 그립니다(10장).

## 5. 정제 결과 저장 (선택)

다음 장부터 매번 정제하지 않도록 깨끗한 표를 저장해 두면 편합니다.

```python
df.to_parquet("population_clean.parquet")   # 또는 df.to_csv("population_clean.csv", index=False)
```

---

## 🌐 웹앱에서는?

웹앱은 같은 일을 자바스크립트 함수로 합니다. `index.html` 을 열어 비교해 보세요.

| 우리가 한 일 | 웹앱의 함수 |
|---|---|
| 인코딩 자동 감지 (utf-8 → cp949) | `decodeBuf(buf)` |
| 콤마 숫자 → 숫자 | `num(v)` |
| 괄호 코드 제거·표준화 | `buildRows(table)` |
| 중복·출장소 제거, 시도 분류 | `initData(rows)`, `sidoOf(name)` |

원리는 같습니다. **"읽고 → 숫자로 바꾸고 → 이름 다듬고 → 이상한 행 버리기."**

## 🧪 직접 해보기
1. 정제 전후 행 수가 몇 개 줄었나요? 어떤 행이 사라졌는지 `print` 로 확인해 보세요.
2. `df[df["행정구역"].str.contains("역삼")]` 로 역삼동 관련 행을 찾아보세요.
3. (도전) 인구 열의 합이 0인 행이 있는지 찾아보고, 있다면 왜 그런지 생각해 보세요.

---
다음 → [4장. 탐색적 분석과 지표 만들기](ch04-eda-metrics.md)
