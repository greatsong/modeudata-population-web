import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────────────────
AGES = list(range(101))
MALE_COLS = [f"남자_{a}세" for a in range(100)] + ["남자_100세이상"]
FEMALE_COLS = [f"여자_{a}세" for a in range(100)] + ["여자_100세이상"]
SIDO_ORDER = [
    "서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시",
    "울산광역시", "세종특별자치시", "경기도", "강원특별자치도", "충청북도", "충청남도",
    "전북특별자치도", "전라남도", "경상북도", "경상남도", "제주특별자치도",
]
BLUE, PINK, GOLD, ACCENT = "#4C8DF6", "#F4789B", "#F4B400", "#E8930C"
CLU_COLORS = ["#E8930C", "#4C8DF6", "#54A24B", "#E45756",
              "#B279A2", "#3BA6A0", "#F58518", "#9C755F"]

st.set_page_config(page_title="🏘️ 우리 동네 인구 구조 · Streamlit", page_icon="🏘️", layout="wide")


# ──────────────────────────────────────────────────────────────────────────
# 데이터 로딩 · 정제
# ──────────────────────────────────────────────────────────────────────────
def _read_csv_any(src):
    """utf-8 / cp949 인코딩을 자동으로 시도해 CSV를 읽는다."""
    last = None
    for enc in ("utf-8-sig", "cp949", "utf-8"):
        try:
            if hasattr(src, "seek"):
                src.seek(0)
            return pd.read_csv(src, encoding=enc, thousands=",", dtype=str)
        except (UnicodeDecodeError, Exception) as e:  # noqa: BLE001
            last = e
    raise last


def _sido_of(name: str):
    if name == "대한민국 전체":
        return "전국"
    for s in SIDO_ORDER:
        if name == s or name.startswith(s + " "):
            return s
    return None


@st.cache_data(show_spinner="데이터를 불러오는 중이에요…")
def load_population(file_bytes=None, file_name="기본 데이터"):
    src = io.BytesIO(file_bytes) if file_bytes is not None else "population_2026_05.csv"
    df = _read_csv_any(src)
    df.columns = [c.replace("﻿", "").strip() for c in df.columns]

    # 인구 열을 숫자로
    for c in MALE_COLS + FEMALE_COLS:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # 행정구역 이름 정리 (괄호 코드 제거) · 코드 컬럼
    df["행정구역"] = (
        df["행정구역"].astype(str)
        .str.replace(r"\s*\(\d+\)\s*$", "", regex=True)
        .str.strip()
    )
    if "코드" in df.columns:
        df["코드"] = df["코드"].astype(str).str.strip()
    else:
        df["코드"] = ""

    # 중복 · 비표준 행 제거
    df = df.drop_duplicates(subset="행정구역").reset_index(drop=True)
    df = df[~df["행정구역"].str.contains("출장소", na=False)].reset_index(drop=True)
    df["sido"] = df["행정구역"].map(_sido_of)
    df = df[df["sido"].notna()].reset_index(drop=True)

    # 전국 합계 행 추가 (17개 시도 행의 합)
    sido_rows = df[df["행정구역"].isin(SIDO_ORDER)]
    total_row = {"행정구역": "대한민국 전체", "코드": "", "sido": "전국"}
    for c in MALE_COLS + FEMALE_COLS:
        total_row[c] = int(sido_rows[c].sum())
    df = pd.concat([pd.DataFrame([total_row]), df], ignore_index=True)

    df = _add_metrics(df)
    return df, file_name


def _add_metrics(df):
    M = df[MALE_COLS].to_numpy(float)
    F = df[FEMALE_COLS].to_numpy(float)
    T = M + F
    ages = np.array(AGES)
    total = T.sum(1)
    safe = np.clip(total, 1, None)

    df["총인구"] = total.astype(int)
    df["남자수"] = M.sum(1).astype(int)
    df["여자수"] = F.sum(1).astype(int)
    df["평균연령"] = (T * ages).sum(1) / safe
    df["고령화율"] = T[:, 65:].sum(1) / safe * 100
    df["유소년비율"] = T[:, :15].sum(1) / safe * 100
    df["소멸위험지수"] = F[:, 20:40].sum(1) / np.clip(T[:, 65:].sum(1), 1, None)
    df["노령화지수"] = T[:, 65:].sum(1) / np.clip(T[:, :15].sum(1), 1, None) * 100
    df["노년부양비"] = T[:, 65:].sum(1) / np.clip(T[:, 15:65].sum(1), 1, None) * 100

    cum = np.cumsum(T, axis=1)
    half = (total / 2)[:, None]
    df["중위연령"] = (cum < half).sum(1).clip(0, 100)
    return df


@st.cache_data(show_spinner=False)
def load_centroids():
    try:
        c = _read_csv_any("dong_centroids.csv")
    except Exception:  # noqa: BLE001
        return {}
    c.columns = [x.replace("﻿", "").strip() for x in c.columns]
    out = {}
    for _, r in c.iterrows():
        out[str(r["코드"]).strip()] = (float(r["lat"]), float(r["lon"]))
    return out


def band(idx):
    if idx < 0.2:
        return "소멸 고위험", "#E45756"
    if idx < 0.5:
        return "소멸위험 진입", "#F58518"
    if idx < 1.0:
        return "소멸 주의", "#E8C13F"
    if idx < 1.5:
        return "보통", "#88C580"
    return "저위험", "#54A24B"


def fmt_pop(n):
    n = int(round(n))
    return f"{n/10000:.1f}만 명" if n >= 10000 else f"{n:,}명"


def row_ages(row):
    m = np.array([row[c] for c in MALE_COLS], float)
    f = np.array([row[c] for c in FEMALE_COLS], float)
    return m, f


# ──────────────────────────────────────────────────────────────────────────
# 사이드바
# ──────────────────────────────────────────────────────────────────────────
st.sidebar.title("🏘️ 우리 동네 인구 구조")
st.sidebar.caption("우리 동네 인구 구조 · Streamlit 판")

up = st.sidebar.file_uploader(
    "📂 다른 달 CSV 올리기 (행정안전부 연령별 인구현황)", type="csv"
)
if up is not None:
    df, src_name = load_population(up.getvalue(), up.name)
    st.sidebar.success(f"업로드: {src_name}")
else:
    df, src_name = load_population()

CENT = load_centroids()

sidos = ["전국"] + [s for s in SIDO_ORDER if (df["sido"] == s).any()]
sido = st.sidebar.selectbox("시 / 도", sidos, index=0)
town_list = df[df["sido"] == sido]["행정구역"].tolist()
town = st.sidebar.selectbox("행정구역 (시군구·동)", town_list, index=0 if town_list else None)

st.sidebar.markdown("---")
st.sidebar.caption("데이터: 2026년 5월 행정안전부 주민등록 인구현황 · 그래프: Plotly")


# ──────────────────────────────────────────────────────────────────────────
# 상단 지표
# ──────────────────────────────────────────────────────────────────────────
row = df[df["행정구역"] == town].iloc[0]
st.title(f"🏘️ {town}")
st.caption("연령·성별 인구로 동네의 ‘나이 지도’를 그려 봅니다.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("총인구", fmt_pop(row["총인구"]))
c2.metric("평균 연령", f"{row['평균연령']:.1f}세")
c3.metric("고령화율(65+)", f"{row['고령화율']:.1f}%")
c4.metric("유소년(0–14)", f"{row['유소년비율']:.1f}%")
bl, bc = band(row["소멸위험지수"])
c5.metric("소멸위험지수", f"{row['소멸위험지수']:.2f}", bl)

st.caption(
    f"남자 {row['남자수']:,}명 · 여자 {row['여자수']:,}명 · "
    f"중위연령 {int(row['중위연령'])}세 · 노령화지수 {row['노령화지수']:.0f} · 노년부양비 {row['노년부양비']:.0f}"
)

# ──────────────────────────────────────────────────────────────────────────
# 탭
# ──────────────────────────────────────────────────────────────────────────
tab_pyr, tab_dist, tab_cmp, tab_twin, tab_clu, tab_map = st.tabs(
    ["🔺 인구 피라미드", "📈 연령 분포", "🆚 동네 비교", "👯 쌍둥이 동네", "🧩 군집 분석", "🗺️ 지도"]
)

# ── 인구 피라미드 ──
with tab_pyr:
    m, f = row_ages(row)
    lim = max(m.max(), f.max(), 1) * 1.1
    fig = go.Figure()
    fig.add_bar(y=AGES, x=-m, orientation="h", name="남자", marker_color=BLUE,
                customdata=m, hovertemplate="%{y}세 · 남 %{customdata:,.0f}명<extra></extra>")
    fig.add_bar(y=AGES, x=f, orientation="h", name="여자", marker_color=PINK,
                hovertemplate="%{y}세 · 여 %{x:,.0f}명<extra></extra>")
    fig.update_layout(barmode="relative", bargap=0.12, height=720,
                      legend=dict(orientation="h", y=1.04),
                      margin=dict(l=40, r=20, t=30, b=40))
    fig.update_xaxes(title="인구 수", range=[-lim, lim],
                     tickvals=[-lim / 2, 0, lim / 2],
                     ticktext=[f"{int(lim/2):,}", "0", f"{int(lim/2):,}"])
    fig.update_yaxes(title="나이(세)", dtick=10)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("왼쪽이 남자, 오른쪽이 여자, 한 칸이 1살. 위가 넓으면 나이 든 동네, 아래가 넓으면 젊은 동네예요.")

# ── 연령 분포 ──
with tab_dist:
    m, f = row_ages(row)
    t = m + f
    peak = int(t.argmax())
    low = int(t[:100].argmin())
    fig = go.Figure()
    fig.add_scatter(x=AGES, y=t, mode="lines", fill="tozeroy", line_color=ACCENT,
                    fillcolor="rgba(244,180,0,0.25)", name="전체",
                    hovertemplate="%{x}세 · %{y:,.0f}명<extra></extra>")
    fig.add_scatter(x=[peak, low], y=[t[peak], t[low]], mode="markers+text",
                    text=[f"최다 {peak}세", f"최소 {low}세"], textposition="top center",
                    marker=dict(size=11, color=["#E45756", BLUE]), showlegend=False, hoverinfo="skip")
    fig.update_layout(height=440, margin=dict(l=55, r=20, t=20, b=45),
                      xaxis_title="나이(세)", yaxis_title="인구 수")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"👴 인구가 가장 많은 나이: {peak}세 ({t[peak]:,.0f}명) · "
               f"👶 가장 적은 나이(100세 미만): {low}세 ({t[low]:,.0f}명)")

# ── 동네 비교 ──
with tab_cmp:
    opts = df[df["sido"] == sido]["행정구역"].tolist()
    cc1, cc2, cc3 = st.columns([2, 2, 1])
    a = cc1.selectbox("동네 A", opts, index=0, key="cmpA")
    b = cc2.selectbox("동네 B", opts, index=min(1, len(opts) - 1), key="cmpB")
    norm = cc3.checkbox("비율(%)로", value=True)

    def line(name):
        r = df[df["행정구역"] == name].iloc[0]
        mm, ff = row_ages(r)
        t = mm + ff
        return t / max(t.sum(), 1) * 100 if norm else t

    fig = go.Figure()
    fig.add_scatter(x=AGES, y=line(a), mode="lines", name=a, line_color=ACCENT)
    fig.add_scatter(x=AGES, y=line(b), mode="lines", name=b, line_color=BLUE)
    fig.update_layout(height=440, margin=dict(l=55, r=20, t=20, b=45),
                      legend=dict(orientation="h", y=1.1),
                      xaxis_title="나이(세)", yaxis_title="비율(%)" if norm else "인구 수")
    st.plotly_chart(fig, use_container_width=True)

# ── 쌍둥이 동네 (코사인 유사도) ──
@st.cache_data(show_spinner=False)
def leaf_matrix(src_name):
    leaf = df[df["행정구역"].str.endswith(("동", "읍", "면"))].reset_index(drop=True)
    T = leaf[MALE_COLS].to_numpy(float) + leaf[FEMALE_COLS].to_numpy(float)
    R = T / np.clip(T.sum(1, keepdims=True), 1, None)
    return leaf, R

with tab_twin:
    st.caption("연령·성별 인구의 ‘모양’이 가장 비슷한 동네를 찾아요(코사인 유사도). 읍·면·동끼리 비교합니다.")
    leaf, R = leaf_matrix(src_name)
    if town in set(leaf["행정구역"]):
        ti = leaf.index[leaf["행정구역"] == town][0]
        tv = R[ti]
        sims = (R @ tv) / (np.linalg.norm(R, axis=1) * np.linalg.norm(tv) + 1e-12)
        order = np.argsort(-sims)
        order = [i for i in order if i != ti][:5]
        tbl = pd.DataFrame({
            "행정구역": leaf.loc[order, "행정구역"].values,
            "닮은 정도": [f"{sims[i]*100:.1f}%" for i in order],
            "평균연령": [f"{leaf.loc[i, '평균연령']:.1f}세" for i in order],
            "소멸위험지수": [f"{leaf.loc[i, '소멸위험지수']:.2f}" for i in order],
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        fig = go.Figure()
        fig.add_scatter(x=AGES, y=R[ti] * 100, mode="lines", name="⭐ " + town,
                        line=dict(width=3, color=ACCENT))
        for i in order:
            fig.add_scatter(x=AGES, y=R[i] * 100, mode="lines", name=leaf.loc[i, "행정구역"], opacity=0.8)
        fig.update_layout(height=420, margin=dict(l=55, r=20, t=20, b=45),
                          legend=dict(orientation="h", y=1.15),
                          xaxis_title="나이(세)", yaxis_title="비율(%)")
        st.plotly_chart(fig, use_container_width=True)

        # 쌍둥이 지도
        pts = [(leaf.loc[i, "행정구역"], CENT.get(str(leaf.loc[i, "코드"]))) for i in order]
        pts = [(n, c) for n, c in pts if c]
        sel = CENT.get(str(leaf.loc[ti, "코드"]))
        if pts:
            mfig = go.Figure()
            mfig.add_scattermapbox(lat=[c[0] for _, c in pts], lon=[c[1] for _, c in pts],
                                   mode="markers", marker=dict(size=12, color=BLUE),
                                   text=[n for n, _ in pts], hoverinfo="text", name="쌍둥이")
            if sel:
                mfig.add_scattermapbox(lat=[sel[0]], lon=[sel[1]], mode="markers",
                                       marker=dict(size=18, color=ACCENT), text=["⭐ " + town],
                                       hoverinfo="text", name="선택")
            lats = [c[0] for _, c in pts] + ([sel[0]] if sel else [])
            lons = [c[1] for _, c in pts] + ([sel[1]] if sel else [])
            span = max(max(lats) - min(lats), max(lons) - min(lons))
            zoom = 5.2 if span > 3 else 7 if span > 1 else 9 if span > 0.3 else 11
            mfig.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0),
                               mapbox=dict(style="carto-positron",
                                           center=dict(lat=np.mean(lats), lon=np.mean(lons)), zoom=zoom),
                               legend=dict(orientation="h", y=1.0))
            st.plotly_chart(mfig, use_container_width=True)
    else:
        st.info("읍·면·동을 선택하면 쌍둥이 동네를 찾아 줘요. (현재 선택은 시도/시군구 단위예요)")

# ── 군집 분석 (K-평균 + PCA) ──
with tab_clu:
    st.caption("동네(읍·면·동)를 인구 구조의 모양으로 비슷한 것끼리 묶어요(K-평균). "
               "연령별(101차원)·성별×연령별(202차원) 중 고르고, 인구 규모까지 반영할 수 있어요.")
    g1, g2, g3, g4 = st.columns(4)
    cscope = g1.selectbox("대상 범위", sidos, index=0, key="cscope")
    cmode = g2.selectbox("차원(특징)", ["연령별 (101차원)", "성별·연령별 (202차원)"], key="cmode")
    cK = g3.slider("군집 수 K", 2, 8, 4)
    csize = g4.checkbox("인구 규모도 반영", value=False)

    pool = df[df["행정구역"].str.endswith(("동", "읍", "면"))]
    if cscope != "전국":
        pool = pool[pool["sido"] == cscope]
    pool = pool.reset_index(drop=True)

    if len(pool) < cK:
        st.warning(f"이 범위의 동네가 {len(pool)}개뿐이라 K={cK} 군집을 만들 수 없어요. 범위를 넓히거나 K를 줄여 보세요.")
    else:
        M = pool[MALE_COLS].to_numpy(float)
        F = pool[FEMALE_COLS].to_numpy(float)
        tot = (M + F).sum(1, keepdims=True)
        if cmode.startswith("성별"):
            X = np.hstack([M / np.clip(tot, 1, None), F / np.clip(tot, 1, None)])
        else:
            T = M + F
            X = T / np.clip(tot, 1, None)
        if csize:
            X = np.hstack([X, np.log10(tot + 1)])

        with st.spinner("군집화 계산 중…"):
            Xs = StandardScaler().fit_transform(X)
            labels = KMeans(n_clusters=cK, n_init=10, random_state=42).fit_predict(Xs)
            P = PCA(n_components=2, random_state=42).fit_transform(Xs)
        pool = pool.assign(군집=labels, PC1=P[:, 0], PC2=P[:, 1])

        dim_txt = "성별·연령별 202차원" if cmode.startswith("성별") else "연령별 101차원"
        st.success(f"{cscope} 읍·면·동 {len(pool):,}개를 {dim_txt}{' + 인구 규모' if csize else ''} "
                   f"기준으로 {cK}개 군집으로 나눴어요.")

        # 엘보우 방법 — 적당한 K 찾기
        ks = list(range(2, 11))
        with st.spinner("엘보우 계산 중…"):
            inertias = [KMeans(n_clusters=k, n_init=5, random_state=42).fit(Xs).inertia_ for k in ks]
        efig = go.Figure()
        efig.add_scatter(x=ks, y=inertias, mode="lines+markers",
                         line=dict(color=ACCENT, width=2), marker=dict(size=8, color=ACCENT),
                         hovertemplate="K=%{x} · 관성 %{y:,.0f}<extra></extra>")
        efig.add_scatter(x=[cK], y=[inertias[ks.index(cK)]], mode="markers",
                         marker=dict(size=15, color="#E45756", line=dict(color="#fff", width=2)),
                         hovertemplate="지금 K=%{x}<extra></extra>", showlegend=False)
        efig.update_layout(height=300, margin=dict(l=60, r=20, t=20, b=42), showlegend=False,
                           xaxis=dict(title="군집 수 K", dtick=1), yaxis_title="관성(inertia)")
        st.subheader("📐 군집 수 정하기 (엘보우 방법)")
        st.plotly_chart(efig, use_container_width=True)
        st.caption("K를 늘릴수록 군집은 빽빽해지지만(관성↓), 어느 지점부터 거의 안 줄어요. "
                   "‘팔꿈치’처럼 꺾이는 K가 적당해요. 빨간 점이 지금 고른 K예요.")

        # 산점도
        sfig = go.Figure()
        for c in range(cK):
            sub = pool[pool["군집"] == c]
            sfig.add_scattergl(
                x=sub["PC1"], y=sub["PC2"], mode="markers", name=f"군집 {c+1} ({len(sub)})",
                marker=dict(size=6, color=CLU_COLORS[c % 8], opacity=0.72),
                text=[f"{n}<br>총인구 {fmt_pop(t)} · 평균 {a:.1f}세"
                      for n, t, a in zip(sub["행정구역"], sub["총인구"], sub["평균연령"])],
                hoverinfo="text")
        sfig.update_layout(height=460, margin=dict(l=50, r=20, t=34, b=46),
                           legend=dict(orientation="h", y=1.12),
                           xaxis_title="주성분 1 (PC1)", yaxis_title="주성분 2 (PC2)")
        st.plotly_chart(sfig, use_container_width=True)

        # 군집별 평균 연령 곡선
        pfig = go.Figure()
        for c in range(cK):
            sub = pool[pool["군집"] == c]
            T = sub[MALE_COLS].to_numpy(float) + sub[FEMALE_COLS].to_numpy(float)
            prof = (T / np.clip(T.sum(1, keepdims=True), 1, None)).mean(0) * 100
            pfig.add_scatter(x=AGES, y=prof, mode="lines", name=f"군집 {c+1}",
                             line=dict(color=CLU_COLORS[c % 8], width=2))
        pfig.update_layout(height=360, margin=dict(l=55, r=20, t=30, b=46),
                           legend=dict(orientation="h", y=1.16),
                           xaxis_title="나이(세)", yaxis_title="평균 연령 비율(%)")
        st.subheader("📈 군집별 평균 연령 분포")
        st.plotly_chart(pfig, use_container_width=True)

        # 요약 표 (대표 동네 = 군집 중심 최근접)
        rows = []
        for c in range(cK):
            idx = np.where(labels == c)[0]
            center = Xs[idx].mean(0)
            rep = idx[np.argmin(((Xs[idx] - center) ** 2).sum(1))]
            sub = pool.iloc[idx]
            rows.append({
                "군집": f"군집 {c+1}", "동네 수": len(idx),
                "평균연령": f"{sub['평균연령'].mean():.1f}세",
                "평균 총인구": fmt_pop(sub["총인구"].mean()),
                "평균 소멸위험": f"{sub['소멸위험지수'].mean():.2f}",
                "대표 동네": pool.iloc[rep]["행정구역"],
            })
        st.subheader("📋 군집 요약")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # 군집 지도
        mp = pool.copy()
        mp["latlon"] = mp["코드"].astype(str).map(CENT)
        mp = mp[mp["latlon"].notna()]
        if len(mp):
            cmfig = go.Figure()
            for c in range(cK):
                sub = mp[mp["군집"] == c]
                if not len(sub):
                    continue
                cmfig.add_scattermapbox(
                    lat=[ll[0] for ll in sub["latlon"]], lon=[ll[1] for ll in sub["latlon"]],
                    mode="markers", name=f"군집 {c+1}",
                    marker=dict(size=8, color=CLU_COLORS[c % 8], opacity=0.8),
                    text=sub["행정구역"], hoverinfo="text")
            lats = [ll[0] for ll in mp["latlon"]]
            lons = [ll[1] for ll in mp["latlon"]]
            cmfig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0),
                                legend=dict(orientation="h", y=1.0),
                                mapbox=dict(style="carto-positron",
                                            center=dict(lat=np.mean(lats), lon=np.mean(lons)),
                                            zoom=5.5 if cscope == "전국" else 9))
            st.subheader("🗺️ 군집 지도")
            st.plotly_chart(cmfig, use_container_width=True)

# ── 지도 ──
with tab_map:
    metric = st.radio("점 색깔로 볼 지표", ["소멸위험지수", "평균연령", "총인구"], horizontal=True)
    pool = df[df["행정구역"].str.endswith(("동", "읍", "면"))].copy()
    if sido != "전국":
        pool = pool[pool["sido"] == sido]
    pool["latlon"] = pool["코드"].astype(str).map(CENT)
    pool = pool[pool["latlon"].notna()]
    if not len(pool):
        st.info("중심점(위경도) 데이터를 찾지 못했어요.")
    else:
        col = {"소멸위험지수": "소멸위험지수", "평균연령": "평균연령", "총인구": "총인구"}[metric]
        scale = {"소멸위험지수": "RdYlGn", "평균연령": "RdBu_r", "총인구": "Viridis"}[metric]
        lats = [ll[0] for ll in pool["latlon"]]
        lons = [ll[1] for ll in pool["latlon"]]
        fig = go.Figure(go.Scattermapbox(
            lat=lats, lon=lons, mode="markers",
            marker=dict(size=8, opacity=0.82, color=pool[col], colorscale=scale,
                        showscale=True, colorbar=dict(thickness=12, len=0.8)),
            text=[f"{n}<br>{metric} {v:.2f}" if metric != "총인구" else f"{n}<br>총인구 {fmt_pop(v)}"
                  for n, v in zip(pool["행정구역"], pool[col])],
            hoverinfo="text"))
        fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0),
                          mapbox=dict(style="carto-positron",
                                      center=dict(lat=np.mean(lats), lon=np.mean(lons)),
                                      zoom=5.5 if sido == "전국" else 9.5))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"점 하나가 읍·면·동({len(pool):,}개)이에요. 색이 진할수록 값이 높아요.")

st.markdown("---")
st.caption("데이터: 2026년 5월 행정안전부 주민등록 인구현황 · Streamlit 판")
