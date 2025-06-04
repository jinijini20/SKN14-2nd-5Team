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


# Page configuration
st.set_page_config(
    page_title="데이터 사피엔스",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# 이것은 헤더입니다. 이것은 *매우* 멋진 앱입니다!"
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
        ["개요", "예측", "분석", "성능 지표"]
    )

    st.markdown("---")
    st.markdown("### 🎯 모델 성능")
    st.metric("정확도", "0.881", delta="0.04")
    st.metric("F1 점수", "0.933", delta="0.02")
    st.metric("ROC AUC", "0.794", delta="0.228")

# Main Dashboard Content
if dashboard_mode == "개요":
    st.markdown(
        """
        ## 프로젝트 개요:
        - "데이터 사피엔스" 대시보드는 2016년부터 2018년까지 100,000개의 주문을 포함하는 Olist의 이커머스 주문 공개 데이터셋에서 도출한 인사이트를 보여줍니다.
        - Catboost와 로지스틱 회귀 등의 머신러닝 모델을 활용하여 주문 상태, 체크아웃, 고객 리뷰 등 고객 여정의 다양한 측면에 대한 귀중한 정보를 제공합니다.
        - 데이터셋은 Olist에 제품을 등록한 판매자들의 세부 정보와 고객 행동 및 인구 통계 데이터를 포함합니다.

        ---

        ### 사용된 모델:
        1. **리뷰 점수 예측:**
           - 모델: 로지스틱 회귀
           - 설명: 다양한 요인을 기반으로 고객 리뷰 점수를 예측합니다.

        2. **배송 시간 예측:**
           - 모델: Catboost
           - 설명: Olist에서 주문한 상품의 배송 시간을 예측합니다.

        ---

        ### 모델 평가:
        - **리뷰 점수 예측:**
          - 평가 지표: 정확도, ROC Auc, F1 점수
          - 성능: 정확도 0.881%, ROC Auc 점수 0.794, F1 점수 0.933%를 달성했습니다.

        - **배송 시간 예측:**
          - 평가 지표: RMSE, R2 점수
          - 성능: RMSE 1.176, R2 점수 0.983을 달성했습니다.

        ---
        """, unsafe_allow_html=True)

elif dashboard_mode == "예측":
    st.markdown("<h3 style='text-align: center;'>📊 모델 성능 지표</h3>", unsafe_allow_html=True)

    # 가운데 정렬을 위한 여백 + 3개의 컬럼
    col_left, col1, col2, col3, col_right = st.columns([1, 2, 2, 2, 1])  # 좌우 여백 주기

    with col1:
        st.metric(
            label="🎯 Accuracy",
            value="0.881",
            delta="0.04"
        )

    with col2:
        st.metric(
            label="📊 F1 Score",
            value="0.933",
            delta="0.02"
        )

    with col3:
        st.metric(
            label="📈 ROC AUC",
            value="0.794",
            delta="0.228"
        )

    st.markdown("---")

    # Charts Row
    col1, col2 = st.columns(2)

    # 모델 불러오기
    model = load_model()

    with col1:
        st.subheader("📈 Model Prediction Center")

        if model is None:
            st.error("Model not loaded. Cannot make predictions.")
        else:
            features = ['Frequency', 'Monetary',
                        'delay_days', 'total_days', 'approval_days',
                        'review_flag', 'review_length', 'order_status_binary', 'category_num']

            input_values = {}
            for feature in features:
                input_values[feature] = st.number_input(f"{feature}", value=0.0)

            if st.button('🔮 Predict'):
                try:
                    input_df = pd.DataFrame([input_values], columns=features)
                    prediction = model.predict(input_df)[0]

                    if prediction == 1:
                        st.success("🤩 Customer will be SATISFIED!")
                        st.balloons()
                    else:
                        st.error("😡 Customer will be UNSATISFIED")

                    st.metric("Prediction Result", prediction)

                except Exception as e:
                    st.error(f"Prediction error: {e}")



elif dashboard_mode == "분석":
    st.markdown("### 📊 비즈니스 분석")

    tab1, tab2, tab3 = st.tabs(["매출 분석", "지역별 분포", "제품 성과"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Sample sales data
            months = ['1월', '2월', '3월', '4월', '5월', '6월']
            sales = [45000, 52000, 48000, 61000, 55000, 67000]

            fig = px.bar(x=months, y=sales, title="월별 매출 수익 (R$)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Sample category data
            categories = ['전자제품', '패션', '홈용품', '스포츠', '도서']
            values = [25, 35, 20, 15, 5]

            fig = px.pie(values=values, names=categories, title="카테고리별 매출")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### 🗺️ 브라질 주별 매출")

        # Sample geographic data
        states = ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'GO', 'PE', 'CE']
        sales_by_state = np.random.randint(1000, 15000, len(states))

        fig = px.bar(x=states, y=sales_by_state, title="주별 주문 수")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("#### 📦 제품 성과 지표")

        # Sample product data
        products = ['제품 A', '제품 B', '제품 C', '제품 D', '제품 E']
        ratings = [4.5, 3.8, 4.2, 4.7, 3.9]
        sales_vol = [1200, 800, 1500, 900, 1100]

        fig = px.scatter(x=sales_vol, y=ratings, text=products,
                         title="제품 성과: 매출 vs 평점",
                         labels={'x': '매출량', 'y': '평균 평점'})
        fig.update_traces(textposition='top center')
        st.plotly_chart(fig, use_container_width=True)

elif dashboard_mode == "성능 지표":
    st.markdown("### 📈 모델 및 비즈니스 성과")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 모델 지표")
        metrics_data = {
            '지표': ['정확도', '정밀도', '재현율', 'F1-점수', 'ROC-AUC'],
            '점수': [0.881, 0.875, 0.892, 0.933, 0.794]
        }
        metrics_df = pd.DataFrame(metrics_data)

        fig = px.bar(metrics_df, x='지표', y='점수',
                     title="모델 성능 지표",
                     color='점수', color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📊 비즈니스 KPI")

        # Sample KPI data
        kpi_data = {
            'KPI': ['고객 유지율', '주문 이행률', '정시 배송률', '반품률'],
            '현재': [85, 92, 78, 12],
            '목표': [90, 95, 85, 8]
        }
        kpi_df = pd.DataFrame(kpi_data)

        fig = px.bar(kpi_df, x='KPI', y=['현재', '목표'],
                     title="KPI 성과 vs 목표",
                     barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    # Detailed metrics table
    st.markdown("#### 📋 상세 성과 보고서")

    performance_data = {
        '카테고리': ['고객 만족도', '배송 성과', '매출 성장률', '제품 품질'],
        '이번 달': ['88.1%', '78.5%', '12.3%', '4.2/5.0'],
        '지난 달': ['84.2%', '76.1%', '8.7%', '4.0/5.0'],
        '변화': ['+3.9%', '+2.4%', '+3.6%', '+0.2'],
        '상태': ['✅ 좋음', '⚠️ 개선 필요', '✅ 우수', '✅ 좋음']
    }

    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True)

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