# 🏘️ 우리 동네 인구 구조 (웹)

전국 주민등록 인구로 동네의 인구 구조를 살펴보는 GitHub Pages 프로젝트입니다.
2026년 5월 행정안전부 주민등록 인구(전국 3,918개 행정구역)로
인구 피라미드·연령 분포·동네 비교·**쌍둥이 동네 찾기**·**군집 분석(K-평균)**·소멸위험지수를 봅니다.

군집 분석은 동네를 연령별(101차원)·성별×연령별(202차원) 인구 구조로 K개 군집으로 나누고,
원하면 인구 규모까지 차원에 더해 PCA 2차원 산점도·군집별 연령 곡선·지도로 시각화합니다.

- 라이브: https://greatsong.github.io/modeudata-population-web/
- 데이터: 행정안전부 연령별 인구현황(2026.5)
- 그래프: Plotly.js (클라이언트 사이드, 서버 불필요)

## 📚 입문 교재 — 바이브 코딩으로 데이터 분석 → 머신러닝

**코드 한 줄 안 쓰고, AI에게 시켜서** 이 웹앱을 단계별로 만들어 가는 HTML 입문 교재를 [`docs/`](docs/)에 담았습니다.
pandas·numpy·설치 없이, 복사 버튼이 달린 프롬프트 박스와 시각적 다이어그램으로 실습합니다.

→ **[교재 열기](https://greatsong.github.io/modeudata-population-web/docs/)** (0~10장 + 프롬프트 사전·개념 사전·데이터 안내)

## 🐍 Streamlit 판 (파이썬 앱)

같은 분석(피라미드·연령분포·동네비교·쌍둥이·**군집 분석**·지도)을 **파이썬 Streamlit** 앱으로도 구현했습니다.
pandas·scikit-learn(KMeans·PCA)·Plotly를 사용합니다.

- 앱 코드: [`streamlit_app.py`](streamlit_app.py) · 의존성: [`requirements.txt`](requirements.txt)
- **로컬 실행:** `pip install -r requirements.txt` 후 `streamlit run streamlit_app.py`
- **무료 배포 (Streamlit Community Cloud):**
  1. https://share.streamlit.io 접속 → GitHub 계정으로 로그인
  2. **New app** → 저장소 `greatsong/modeudata-population-web`, 브랜치 `main`, 메인 파일 `streamlit_app.py` 선택
  3. **Deploy** → 잠시 뒤 `https://<앱이름>.streamlit.app` 주소가 생성됩니다.

> GitHub Pages(정적 HTML)와 달리 Streamlit은 파이썬 서버가 필요해 Streamlit Cloud/서버에서 호스팅합니다.
