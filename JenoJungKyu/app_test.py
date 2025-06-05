import alt
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle  # ← 이 줄을 위로 옮김
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_extras.let_it_rain import rain
import altair as alt  # 올바른 방식


# Page configuration
st.set_page_config(
    page_title="데이터 사피엔스",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# 헤더입니다."
    }
)

# Load the best model from GridSearchCV
@st.cache_resource
def load_model():
    try:
        with open('models/best_model.pkl', 'rb') as f:
            data = pickle.load(f)

        # data가 튜플인지 확인 후 처리
        if isinstance(data, tuple):
            # GridSearchCV 튜플 구조에 맞게 조정 필요
            # 예를 들어 첫번째가 GridSearchCV 객체라면
            gs = data[0]
            model = getattr(gs, 'best_estimator_', gs)  # best_estimator_ 있으면 가져오기, 없으면 gs 사용
        else:
            model = data

        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None



# 로딩된 모델 사용
model = load_model()


# Expected feature columns based on your model training

FEATURE_COLUMNS = [
    'Frequency',
    'Monetary',
    'delay_days',
    'total_days',
    'approval_days',
    'review_flag',
    'review_length',
    'order_status_binary',
    'category_num',
]

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .prediction-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .sidebar-section {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="https://i.hizliresim.com/c3v6sx3.png" alt="Olist Logo" width="200"/>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<div class="main-header">
    <h1>🛒 Olist 이커머스 분석 대시보드</h1>
    <p>브라질 이커머스 고객 만족도 및 비즈니스 인텔리전스</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 대시보드 네비게이션")
    dashboard_mode = st.selectbox(
        "대시보드 모드 선택",
        ["개요", "예측", "분석"]
    )

    st.markdown("---")
    st.markdown("### 🎯 모델 성능")
    st.metric("정확도", "0.73", delta="0.04")
    st.metric("F1 점수", "0.55", delta="0.02")
    st.metric("ROC AUC", "0.8", delta="0.228")

# Main Dashboard Content
if dashboard_mode == "개요":
    st.markdown(
        """
        ### 📝 프로젝트 소개
본 프로젝트는 브라질의 대표적인 이커머스 플랫폼 Olist의 공개 데이터를 활용하여 고객의 구매 행동 분석, 재구매 여부 예측, 그리고 이탈 가능성 탐지 등을 수행함으로써, 고객 관리 전략 수립에 기여하고자 진행되었습니다.

### 🔍 프로젝트 필요성
- 최근 이커머스 시장의 경쟁이 심화되면서 기존 고객 유지가 점점 더 중요해지고 있습니다.
- 데이터를 기반으로 고객의 행동을 분석하고 이탈 가능성을 사전에 예측함으로써, 맞춤형 마케팅 전략 수립이 가능합니다.
- 상품의 가격, 무게, 부피 등 제품 특성과 고객 행동 간의 관계를 분석함으로써, 제품 구성 전략 및 물류 최적화에 도움이 됩니다.
  
### 🎯 프로젝트 목표
- 고객의 재구매 여부를 정의하고 분류하는 기준을 수립합니다.

- 고객의 이탈을 판단하는 기준을 설정하고 이에 따라 분석을 진행합니다.

- 제품 속성(가격, 무게, 부피,배송송 등)과 고객의 행동 간 상관관계를 분석합니다.

- 분석 결과를 시각화하고 인사이트를 도출하여 실질적인 개선 방안을 제시합니다.

### 🎈 프로젝트 기대효과
-재구매 고객의 특징을 파악하여 고객 충성도를 높이는 전략 수립이 가능합니다.
- 이탈 가능성이 높은 고객을 조기 식별하여 사전에 대응할 수 있습니다.
- 상품 가격대별 이탈률 및 무게·부피 등의 물류 관련 특성 분석을 통해 제품 전략 및 배송 정책 개선이 기대됩니다.

## 📊 데이터 소개

해당 프로젝트에 사용된 데이터는 브라질 이커머스 플랫폼 Olist의 약 10만 건 이상의 주문 데이터를 포함한 공개 데이터셋입니다

### Olist  데이터
| 테이블명               | 주요 컬럼                                                              | 설명             |
| ------------------ | ------------------------------------------------------------------ | -------------- |
| `orders_df`        | order\_id, customer\_id, order\_status, order\_purchase\_timestamp | 고객의 주문 정보      |
| `customers_df`     | customer\_id, customer\_unique\_id                                 | 고객 고유 식별 정보    |
| `order_items_df`   | order\_id, product\_id, price, freight\_value 등                    | 주문 내 포함된 상품 정보 |
| `products_df`      | product\_id, product\_category\_name, product\_weight\_g 등         | 제품 관련 상세 정보    |
| `order_reviews_df` | order\_id, review\_score                                           | 고객 리뷰 및 만족도 정보 |


---

## 🛠️ 기술 스택
- **언어**
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)

- **데이터 분석**
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243?logo=numpy)

- **머신러닝**
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-EC0000?logo=xgboost)
![LightGBM](https://img.shields.io/badge/LightGBM-Boosting-9ACD32?logo=lightgbm)

- **데이터 시각화**
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?logo=matplotlib)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-00CED1?logo=seaborn)

- **데이터 균형화**
![SMOTE](https://img.shields.io/badge/SMOTE-Data%20Balancing-FF69B4)

- **모델 해석**
![SHAP](https://img.shields.io/badge/SHAP-Model%20Explainability-FF4500)

- **개발 환경**
![Google Colab](https://img.shields.io/badge/Google%20Colab-Cloud-F9AB00?logo=googlecolab)
![VS Code](https://img.shields.io/badge/VS%20Code-IDE-007ACC?logo=visualstudiocode)

- **비전 관리**
![Git](https://img.shields.io/badge/Git-Version--Control-F05032?logo=git)
![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)

---

## 분석 방법론
1. **데이터 전처리**
   - 불필요한 컬럼(User_ID, Name) 제거
   - 음수 월소득 데이터 제거
   - 마지막 로그인(Last_Login) 날짜를 현재 기준 경과 일수로 변환
   - 30일 이상 로그인하지 않은 사용자 이탈 식별(month_churn)
   - 범주형 변수 Label Encoding

2. **데이터 불균형 처리**
   - SMOTE를 활용한 소수 클래스 오버샘플링
   - 학습 데이터와 테스트 데이터 분리(80:20)

3. **특성 표준화**
   - StandardScaler를 통한 수치형 데이터 정규화

4. **모델링 및 하이퍼파라미터 튜닝**
   - 로지스틱 회귀(LogisticRegression)
     - C, penalty, solver 파라미터 튜닝
   - 랜덤 포레스트(RandomForestClassifier)
     - n_estimators, max_depth, min_samples_split, max_features 파라미터 튜닝
   - XGBoost(XGBClassifier)
     - n_estimators, learning_rate, max_depth, subsample, colsample_bytree 파라미터 튜닝
   - LightGBM(LGBMClassifier)
     - num_leaves, learning_rate, n_estimators, feature_fraction 파라미터 튜닝
   - GridSearchCV를 통한 최적 파라미터 탐색.

        ---
        """, unsafe_allow_html=True)

elif dashboard_mode == "예측":
    # 헤더 섹션 - 그라데이션 배경과 함께
    st.markdown("""
        <div style='
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        '>
            <h2 style='color: white; margin: 0; font-weight: 600;'>
                🔮 고객 이탈 예측센터
            </h2>
            <p style='color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
                머신러닝 모델을 활용한 실시간 고객 만족도 예측
            </p>
        </div>
    """, unsafe_allow_html=True)

    # 성능 지표 섹션
    st.markdown("### 📊 모델 성능 지표")

    # 성능 지표를 카드 형태로 표시
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div style='
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem;
                border-radius: 10px;
                text-align: center;
                color: white;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            '>
                <h3 style='margin: 0; font-size: 2.5rem;'>73%</h3>
                <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>🎯 Accuracy</p>
                <small style='opacity: 0.7;'>+4.0% 개선</small>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                padding: 1.5rem;
                border-radius: 10px;
                text-align: center;
                color: white;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            '>
                <h3 style='margin: 0; font-size: 2.5rem;'>80.3%</h3>
                <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>📊 F1 Score</p>
                <small style='opacity: 0.7;'>+2.0% 개선</small>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style='
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                padding: 1.5rem;
                border-radius: 10px;
                text-align: center;
                color: white;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            '>
                <h3 style='margin: 0; font-size: 2.5rem;'>79.4%</h3>
                <p style='margin: 0.5rem 0 0 0; opacity: 0.9;'>📈 ROC AUC</p>
                <small style='opacity: 0.7;'>+22.8% 개선</small>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 예측 섹션 - 전체 중앙 배치
    model = load_model()

    # 중앙 정렬 컨테이너
    st.markdown("""
        <div style='
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border: 1px solid #e1e5e9;
            margin: 2rem auto;
            max-width: 800px;
        '>
            <h3 style='color: #2c3e50; margin-bottom: 1.5rem; text-align: center;'>
                📋 고객 정보 입력
            </h3>
        </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("⚠️ 모델을 불러올 수 없습니다. 관리자에게 문의하세요.")
    else:
        # 입력 필드들을 한 번에 볼 수 있게 배치
        features = ['Frequency', 'Monetary',
                    'delay_days', 'total_days', 'approval_days',
                    'review_flag', 'review_length', 'order_status_binary', 'category_num']

        input_values = {}

        # 3개씩 3줄로 배치
        col1, col2, col3 = st.columns(3)

        with col1:
            input_values['Frequency'] = st.number_input(
                "🔄 구매 빈도",
                value=0.0,
                help="고객의 총 구매 횟수",
                min_value=0.0
            )
            input_values['delay_days'] = st.number_input(
                "⏳ 배송 지연 일수",
                value=0.0,
                help="예정 배송일 대비 지연된 일수",
                min_value=0.0
            )
            input_values['review_flag'] = st.selectbox(
                "📝 리뷰 작성 여부",
                [0, 1],
                help="0: 미작성, 1: 작성"
            )

        with col2:
            input_values['Monetary'] = st.number_input(
                "💰 구매 금액",
                value=0.0,
                help="총 구매 금액",
                min_value=0.0
            )
            input_values['total_days'] = st.number_input(
                "📅 총 소요 일수",
                value=0.0,
                help="주문부터 완료까지 총 일수",
                min_value=0.0
            )
            input_values['review_length'] = st.number_input(
                "📏 리뷰 길이",
                value=0.0,
                help="작성된 리뷰의 문자 수",
                min_value=0.0
            )

        with col3:
            input_values['category_num'] = st.number_input(
                "🏷️ 카테고리 번호",
                value=0.0,
                help="상품 카테고리 분류 번호",
                min_value=0.0
            )
            input_values['approval_days'] = st.number_input(
                "✅ 승인 소요 일수",
                value=0.0,
                help="주문 승인까지 걸린 일수",
                min_value=0.0
            )
            input_values['order_status_binary'] = st.selectbox(
                "📦 주문 상태",
                [0, 1],
                help="0: 미완료, 1: 완료"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # 예측 버튼을 중앙에 크게 배치
        col_center = st.columns([1, 2, 1])
        with col_center[1]:
            predict_button = st.button(
                '🚀 이탈 위험도 예측하기',
                use_container_width=True,
                type="primary"
            )

        if predict_button:
            try:
                input_df = pd.DataFrame([input_values], columns=features)
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None

                st.markdown("<br>", unsafe_allow_html=True)

                if prediction == 1:
                    st.markdown("""
                        <div style='
                            background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
                            padding: 2rem;
                            border-radius: 15px;
                            text-align: center;
                            color: white;
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                            margin: 1rem 0;
                        '>
                            <h2 style='margin: 0; font-size: 2rem;'>✅ 안전 고객!</h2>
                            <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;'>
                                이 고객은 이탈 위험이 낮습니다
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown("""
                        <div style='
                            background: linear-gradient(135deg, #ff6b6b 0%, #ffa8a8 100%);
                            padding: 2rem;
                            border-radius: 15px;
                            text-align: center;
                            color: white;
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                            margin: 1rem 0;
                        '>
                            <h2 style='margin: 0; font-size: 2rem;'>⚠️ 이탈 위험!</h2>
                            <p style='margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;'>
                                고객 유지 전략이 필요합니다
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

                # 확률 정보가 있다면 표시
                if prediction_proba is not None:
                    churn_prob = prediction_proba[1] * 100
                    st.markdown(f"""
                        <div style='
                            background: #f8f9fa;
                            padding: 1rem;
                            border-radius: 10px;
                            text-align: center;
                            margin: 1rem 0;
                        '>
                            <h4 style='color: #495057; margin: 0;'>
                                이탈 위험도: {churn_prob:.1f}%
                            </h4>
                        </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ 예측 중 오류가 발생했습니다: {e}")


elif dashboard_mode == "분석":
    st.markdown("### 📊 Business Analytics")

    tab1, tab2, tab3 = st.tabs(["Order", "Delivery", "Review"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            order_counts_by_month = pd.read_csv("assets/order_counts_by_month.csv")

            # churn 값을 시각화용 라벨로 변환
            order_counts_by_month['churn_label'] = order_counts_by_month['churn'].map({0: '재구매', 1: '이탈'})

            # 막대그래프 (Churn=0)
            bar = alt.Chart(order_counts_by_month[order_counts_by_month['churn'] == 0]).mark_bar().encode(
                x=alt.X('year_month:N', title='날짜', sort=sorted(order_counts_by_month['year_month'].unique().tolist())),
                y=alt.Y('order_count:Q', title='주문 수'),
                color=alt.Color('churn_label:N',
                                scale=alt.Scale(domain=['재구매', '이탈'],
                                                range=['#1f77b4', '#d62728']),
                                legend=alt.Legend(title=None)),
                tooltip=['year_month', 'order_count', 'churn_label']
            )

            # 선그래프 (Churn=1)
            line = alt.Chart(order_counts_by_month[order_counts_by_month['churn'] == 1]).mark_line(point=True).encode(
                x='year_month:N',
                y='order_count:Q',
                color=alt.Color('churn_label:N',
                                scale=alt.Scale(domain=['재구매', '이탈'],
                                                range=['#1f77b4', '#d62728']),
                                legend=None),  # 범례는 bar 그래프에만
                tooltip=['year_month', 'order_count', 'churn_label']
            )

            # 레이어링
            chart = (bar + line).properties(
                width=800,
                height=400,
                title='월별 주문 수 추이'
            )

            st.altair_chart(chart, use_container_width=True)

        with col2:
            order_counts_by_category = pd.read_csv("assets/order_counts_by_category.csv")

            # 1. product_category_name_english별 고유 order_id 수 집계
            category_order_counts = (
                order_counts_by_category.groupby('product_category_name_english')['order_id']
                .nunique()
                .reset_index(name='unique_order_count')
            )

            # 2. 상위 10개 추출
            top10 = category_order_counts.sort_values(by='unique_order_count', ascending=False).head(10)

            # 3. 비율(%) 계산
            top10['percent'] = top10['unique_order_count'] / top10['unique_order_count'].sum() * 100

            # 4. 파이차트 (Altair 원형 그래프는 theta 사용)
            pie_chart = alt.Chart(top10).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="unique_order_count", type="quantitative"),
                color=alt.Color(field="product_category_name_english", type="nominal", title="카테고리"),
                tooltip=[
                    alt.Tooltip("product_category_name_english", title="카테고리"),
                    alt.Tooltip("unique_order_count", title="주문 수"),
                    alt.Tooltip("percent", format=".1f", title="비율 (%)")
                ]
            ).properties(
                width=500,
                height=400,
                title="상위 10개 제품 카테고리별 주문 비율"
            )

            # 차트 표시
            st.altair_chart(pie_chart, use_container_width=True)

    with tab2:
        st.markdown("#### 🗺️ Sales by Brazilian States")

        # Sample geographic data
        states = ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'GO', 'PE', 'CE']
        sales_by_state = np.random.randint(1000, 15000, len(states))

        fig = px.bar(x=states, y=sales_by_state, title="Orders by State")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### 📦 Product Performance Metrics")

        # Sample product data
        products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
        ratings = [4.5, 3.8, 4.2, 4.7, 3.9]
        sales_vol = [1200, 800, 1500, 900, 1100]

        fig = px.scatter(x=sales_vol, y=ratings, text=products,
                         title="Product Performance: Sales vs Rating",
                         labels={'x': 'Sales Volume', 'y': 'Average Rating'})
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)
# Footer
st.markdown("---")
html_footer = """
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 8px; margin-top: 2rem;">
    <p style="color: #666; font-size: 14px;">
        데이터 사피엔스 © 2024 | Olist 이커머스 분석 대시보드
    </p>
    <p style="color: #888; font-size: 12px;">
        Streamlit & Best Model로 구축됨 | 브라질 이커머스 데이터 기반
    </p>
</div>
"""

st.markdown(html_footer, unsafe_allow_html=True)