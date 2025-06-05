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
import matplotlib.pyplot as plt
import seaborn as sns

last_df = pd.read_csv('data/model_df.csv')
last_df['RFM_score_bin'] = pd.qcut(last_df['RFM_add_score'], q=5, labels=False)

# Windows용 한글 폰트 지정
plt.rcParams['font.family'] = 'Malgun Gothic'

# 마이너스 기호 정상 표시 (음수 깨짐 방지)

plt.rcParams['axes.unicode_minus'] = False

# delay_days 구간 설정
last_df['delay_group'] = pd.cut(
    last_df['delay_days'],
    bins=[-100, -10, -1, 0, 1, 5, 10, 100],
    labels=['10일 이상 빠름', '1~9일 빠름', '정시 배송', '1일 지연', '2~5일 지연', '6~10일 지연', '10일 이상 지연']
)

# Page configuration
st.set_page_config(
    page_title="5li5li",
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
        with open('models/XGBoost_final.pkl', 'rb') as f:
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

FEATURE_COLUMNS = [
    'Frequency',
    'Monetary',
    'delay_days',
    'total_days',
    'approval_days',
    'review_flag',
    'review_length',
    'review_score',
    'response_time',
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
    st.metric("F1 점수", "0.803", delta="0.02")
    st.metric("ROC AUC", "0.8", delta="0.228")

# Main Dashboard Content
if dashboard_mode == "개요":
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

    st.markdown(
        """
        ### 📝 프로젝트 소개
본 프로젝트는 브라질의 대표적인 이커머스 플랫폼 Olist의 공개 데이터를 활용하여 고객의 구매 행동 분석, 재구매 여부 예측, 그리고 이탈 가능성 탐지 등을 수행함으로써, 고객 관리 전략 수립에 기여하고자 진행되었습니다.

### 🔍 프로젝트 필요성
- 최근 이커머스 시장의 경쟁이 심화되면서 기존 고객 유지가 점점 더 중요해지고 있습니다.
- 데이터를 기반으로 고객의 행동을 분석하고 이탈 가능성을 사전에 예측함으로써, 맞춤형 마케팅 전략 수립이 가능합니다.
- 상품의 가격, 무게, 부피 등 제품 특성과 고객 행동 간의 관계를 분석함으로써, 제품 구성 전략 및 물류 최적화에 도움이 됩니다.

### 📊 Business 예측탭과분석탭의 사용가이드
| Feature               | 설명                       |
| --------------------- | ------------------------ |
| `Frequency`           | 고객의 총 구매 횟수              |
| `Monetary`            | 총 구매 금액                  |
| `delay_days`          | 배송 지연 일수                 |
| `total_days`          | 주문부터 완료까지 소요 일수          |
| `approval_days`       | 주문 승인까지 걸린 일수            |
| `review_flag`         | 리뷰 작성 여부 (0: 미작성, 1: 작성) |
| `review_length`       | 작성한 리뷰의 문자 수             |
| `review_score`        | 리뷰 평점 (1\~5점)            |
| `response_time`       | 고객 문의 응답까지 걸린 시간         |
| `order_status_binary` | 주문 상태 (0: 미완료, 1: 완료)    |
| `category_num`        | 상품 카테고리 번호 (카테고리별 군집화)   |
위의 하이퍼 파라미터 넣고 예측을 누르면 XG부스트로 고객이탈을 예측합니다


### 1️⃣ Order 탭 - 주문 분석
주요 차트

월별 매출 추이: 재구매 고객 vs 이탈 고객의 매출 변화
주문 패턴 분석: 주문수(막대)와 AOV(선그래프) 복합 차트
인기 카테고리: 고객 유형별 상위 5개 카테고리 파이차트

핵심 인사이트

2017년 11월 이후 재구매 고객 매출 급증
서비스 활성화 시점 파악 가능

### #️⃣ Delivery 탭 - 배송 분석
주요 차트

주문 상태별 이탈 분포: 배송 상태에 따른 이탈 고객 비율
배송 지연 이탈률: 정시배송 vs 지연배송 이탈률 비교
배송 소요 기간별 이탈률: 7일 단위로 구간화한 이탈률 추이

핵심 발견

배송 지연 시 이탈률 증가
배송 시간이 고객 유지의 핵심 요소

### Review 탭 - 고객 만족도 분석
주요 기능

핵심 지표 카드: 평균 리뷰점수, 정시배송률, 응답시간, 고충성고객 비율
배송 성과별 만족도: 배송 지연 정도에 따른 리뷰 점수 분포
고객 충성도별 참여도: RFM 점수별 설문 응답 시간 분석

인터랙티브 기능

차트 스타일 선택 (박스플롯/바이올린플롯)
색상 테마 변경
이상치 표시 옵션


### 🎮 사용 방법

탭 선택: 분석하고 싶은 영역의 탭 클릭
차트 상호작용: 차트 위에 마우스를 올려 상세 정보 확인
설정 조정: Review 탭에서 차트 스타일과 색상 커스터마이징
인사이트 확인: 각 섹션의 핵심 발견사항과 통계 정보 검토


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
                help="예정 배송일 대비 지연된 일수"
            )
            input_values['review_flag'] = st.selectbox(
                "📝 리뷰 작성 여부",
                [0, 1],
                help="0: 미작성, 1: 작성"
            )
            input_values['review_score'] = st.slider(
                "⭐ 리뷰 점수",
                min_value=1.0,
                max_value=5.0,
                value=5.0,
                step=0.1,
                help="고객이 남긴 리뷰 평점 (1~5)"
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
            input_values['response_time'] = st.number_input(
                "⏱️ 응답 시간 (초)",
                value=0.0,
                help="설문 응답까지 걸린 시간",
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
                input_df = pd.DataFrame([{k: input_values[k] for k in features}])
                prediction_proba = model.predict_proba(input_df)[0]

                churn_prob = prediction_proba[1]  # 이탈 확률
                threshold = 0.4  # 사용자가 정할 수 있는 기준 값

                st.markdown("<br>", unsafe_allow_html=True)

                if churn_prob >= threshold:
                    st.markdown(f"""
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
                else:
                    st.markdown(f"""
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

                st.markdown(f"""
                    <div style='
                        background: #f8f9fa;
                        padding: 1rem;
                        border-radius: 10px;
                        text-align: center;
                        margin: 1rem 0;
                    '>
                        <h4 style='color: #495057; margin: 0;'>
                            이탈 위험도: {churn_prob * 100:.1f}%
                        </h4>
                        <small style='color: #868e96;'>기준: {threshold * 100:.0f}% 이상이면 이탈로 판단</small>
                    </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ 예측 중 오류가 발생했습니다: {e}")



elif dashboard_mode == "분석":
    st.markdown("### 📊 Business Analytics")

    tab1, tab2, tab3 = st.tabs(["Order", "Delivery", "Review"])

    with tab1:
        # 페이지 헤더

        st.markdown("""
                <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                            border-radius: 10px; margin-bottom: 30px;">
                    <h1 style="color: white; margin: 0; font-size: 2.5em;">📊 매출 및 고객 분석 대시보드</h1>
                    <p style="color: white; margin-top: 10px; font-size: 1.2em;">배송 성과와 고객 충성도 인사이트</p>
                </div>
                """, unsafe_allow_html=True)
        # 1행: 월별 매출 추이
        with st.container():
            st.markdown("#### 💰 월별 매출 추이")

            # 데이터 불러오기
            monthly_stats = pd.read_csv("assets/monthly_stats.csv")
            monthly_stats = monthly_stats.sort_values(['year', 'month'])
            monthly_stats['churn_label'] = monthly_stats['churn'].map({0: '재구매 고객', 1: '이탈 고객'})

            fig_revenue = px.line(
                monthly_stats,
                x='year_month',
                y='total_revenue',
                color='churn_label',
                labels={
                    'total_revenue': '매출 (원)',
                    'churn_label': '고객 유형',
                    'year_month': '기간'
                },
                color_discrete_map={
                    '재구매 고객': '#2E86AB',  # 차분한 파란색
                    '이탈 고객': '#E63946'  # 차분한 빨간색
                }
            )

            fig_revenue.update_layout(
                yaxis_title="매출 (원)",
                xaxis_title="기간",
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(l=20, r=20, t=40, b=20)
            )

            # x축 레이블 회전 및 격자 추가
            fig_revenue.update_xaxes(
                tickangle=45,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'
            )
            fig_revenue.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'
            )

            st.plotly_chart(fig_revenue, use_container_width=True)

        st.markdown("---")

        # 2행: 월별 인당 주문금액 & 주문수 복합 그래프
        with st.container():
            st.markdown("#### 📈 주문 패턴 분석")

            # 메트릭 설명
            col_desc1, col_desc2 = st.columns(2)
            with col_desc1:
                st.markdown("**📊 주문수**: 월별 총 주문 건수")
            with col_desc2:
                st.markdown("**💳 AOV**: 평균 주문 금액 (Average Order Value)")

            # 데이터 불러오기
            order_counts_by_month = pd.read_csv("assets/order_counts_by_month.csv")

            # 복합 그래프 생성
            from plotly.subplots import make_subplots

            fig_combo = make_subplots(
                specs=[[{"secondary_y": True}]],
                subplot_titles=()
            )

            # 비이탈 고객 데이터
            non_churn_stats = monthly_stats[monthly_stats['churn'] == 0]
            non_churn_orders = order_counts_by_month[order_counts_by_month['churn'] == 0]

            # 이탈 고객 데이터
            churn_stats = monthly_stats[monthly_stats['churn'] == 1]
            churn_orders = order_counts_by_month[order_counts_by_month['churn'] == 1]

            # 주문수 (막대그래프) - 1차 y축
            fig_combo.add_trace(
                go.Bar(
                    x=non_churn_orders['year_month'],
                    y=non_churn_orders['order_count'],
                    name='재구매 주문수',
                    marker_color='rgba(46, 134, 171, 0.7)',
                    hovertemplate='<b>재구매 고객</b><br>기간: %{x}<br>주문수: %{y:,}건<extra></extra>'
                ),
                secondary_y=False,
            )

            fig_combo.add_trace(
                go.Bar(
                    x=churn_orders['year_month'],
                    y=churn_orders['order_count'],
                    name='이탈 주문수',
                    marker_color='rgba(230, 57, 70, 0.7)',
                    hovertemplate='<b>이탈 고객</b><br>기간: %{x}<br>주문수: %{y:,}건<extra></extra>'
                ),
                secondary_y=False,
            )

            # 인당 주문금액 (라인그래프) - 2차 y축
            fig_combo.add_trace(
                go.Scatter(
                    x=non_churn_stats['year_month'],
                    y=non_churn_stats['avg_order_value'],
                    mode='lines+markers',
                    name='재구매 AOV',
                    line=dict(color='#2E86AB', width=4),
                    marker=dict(size=10, symbol='circle'),
                    hovertemplate='<b>재구매 고객 AOV</b><br>기간: %{x}<br>금액: ₩%{y:,.0f}<extra></extra>'
                ),
                secondary_y=True,
            )

            fig_combo.add_trace(
                go.Scatter(
                    x=churn_stats['year_month'],
                    y=churn_stats['avg_order_value'],
                    mode='lines+markers',
                    name='이탈 AOV',
                    line=dict(color='#E63946', width=4),
                    marker=dict(size=10, symbol='circle'),
                    hovertemplate='<b>이탈 고객 AOV</b><br>기간: %{x}<br>금액: ₩%{y:,.0f}<extra></extra>'
                ),
                secondary_y=True,
            )

            # y축 레이블 설정
            fig_combo.update_yaxes(title_text="주문 건수", secondary_y=False)
            fig_combo.update_yaxes(title_text="평균 주문금액 (원)", secondary_y=True)

            # 레이아웃 설정
            fig_combo.update_layout(
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(l=20, r=20, t=40, b=20),
                xaxis_title="기간"
            )

            fig_combo.update_xaxes(
                tickangle=45,
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)'
            )
            fig_combo.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                secondary_y=False
            )

            st.plotly_chart(fig_combo, use_container_width=True)

        st.markdown("---")

        # 3행: 카테고리 비율 파이차트
        st.markdown("#### 🏷️ 인기 카테고리 분석")

        # 설명 텍스트
        st.markdown("**고객 유형별 상위 5개 인기 카테고리 비교**")

        col3, col4 = st.columns(2)

        # 데이터 불러오기
        data = pd.read_csv("assets/order_counts_by_category.csv")

        # col3: 재구매 고객 파이차트
        with col3:
            # churn = 0인 데이터 필터링
            churn_0_data = data[data['churn'] == 0]
            category_counts_0 = churn_0_data.groupby('product_category_name_english')[
                'order_id'].nunique().reset_index()
            category_counts_0.columns = ['category', 'order_count']
            top5_categories_0 = category_counts_0.nlargest(5, 'order_count')
            total_orders_0 = top5_categories_0['order_count'].sum()
            top5_categories_0['ratio'] = top5_categories_0['order_count'] / total_orders_0
            top5_categories_0 = top5_categories_0.sort_values('ratio', ascending=False)

            # 파이차트 생성
            fig_0 = px.pie(
                top5_categories_0,
                values='ratio',
                names='category',
                title="<b>🔄 재구매 고객</b>",
                color_discrete_sequence=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
            )

            fig_0.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>비율: %{percent}<br>주문수: %{value:.0%}<extra></extra>',
                direction='clockwise',
                sort=False
            )

            fig_0.update_layout(
                height=400,
                font=dict(size=11),
                showlegend=False,
                title_x=0.5,
                margin=dict(l=20, r=20, t=60, b=20)
            )

            st.plotly_chart(fig_0, use_container_width=True)

        # col4: 이탈 고객 파이차트
        with col4:
            # churn = 1인 데이터 필터링
            churn_1_data = data[data['churn'] == 1]
            category_counts_1 = churn_1_data.groupby('product_category_name_english')[
                'order_id'].nunique().reset_index()
            category_counts_1.columns = ['category', 'order_count']
            top5_categories_1 = category_counts_1.nlargest(5, 'order_count')
            total_orders_1 = top5_categories_1['order_count'].sum()
            top5_categories_1['ratio'] = top5_categories_1['order_count'] / total_orders_1
            top5_categories_1 = top5_categories_1.sort_values('ratio', ascending=False)

            # 파이차트 생성
            fig_1 = px.pie(
                top5_categories_1,
                values='ratio',
                names='category',
                title="<b>⚠️ 이탈 고객</b>",
                color_discrete_sequence=['#E63946', '#F77F00', '#FCBF49', '#EAE2B7', '#D62828']
            )

            fig_1.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>비율: %{percent}<br>주문수: %{value:.0%}<extra></extra>',
                direction='clockwise',
                sort=False
            )

            fig_1.update_layout(
                height=400,
                font=dict(size=11),
                showlegend=False,
                title_x=0.5,
                margin=dict(l=20, r=20, t=60, b=20)
            )

            st.plotly_chart(fig_1, use_container_width=True)

        # 인사이트 섹션
        st.markdown("---")

        with st.expander("💡 **주요 인사이트**", expanded=False):
            col_insight1, col_insight2 = st.columns(2)

            with col_insight1:
                st.markdown("""
                **📈 📌주문 관련 Insight
2017년 11월 이후 재구매 고객 매출이 급증함
2017년 11월을 기점으로 서비스가 활성화 된 것으로 추정됨
                """)



    with tab2:
        st.markdown("""
                <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                            border-radius: 10px; margin-bottom: 30px;">
                    <h1 style="color: white; margin: 0; font-size: 2.5em;">📦 배송 분석 대시보드</h1>
                    <p style="color: white; margin-top: 10px; font-size: 1.2em;">배송 성과와 고객 충성도 인사이트</p>
                </div>
                """, unsafe_allow_html=True)
        # 상단 행: 주문 상태별 분포와 배송 지연 분석
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("##### 📋 주문 상태별 이탈 분포")

            # csv 파일 읽기
            order_churn = pd.read_csv('data/order_churn.csv')

            # 이탈 여부에 따른 주문상태 count 집계
            chart_data = order_churn.groupby(['order_status', 'churn_str']).size().reset_index(name='count')
            sort = ['shipped', 'canceled', 'unavailable', 'processing', 'created', 'approved']

            # Altair 그래프 생성
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('order_status:N', title='배송 상태', sort=sort, axis=alt.Axis(labelAngle=-45)),
                y=alt.Y('count:Q', title='건수'),
                color=alt.Color('churn_str:N', title='이탈 여부',
                                scale=alt.Scale(range=['#2E86AB', '#A23B72']),
                                legend=alt.Legend(orient='top', titleFontSize=12, labelFontSize=11)),
                xOffset='churn_str:N',
                tooltip=['order_status:N', 'churn_str:N', 'count:Q']
            ).properties(
                width='container',
                height=350
            ).resolve_scale(
                color='independent'
            )

            st.altair_chart(chart, use_container_width=True)

        with col2:
            st.markdown("##### ⏰ 배송 지연에 따른 이탈률")

            # csv 파일 읽기
            model_df = pd.read_csv('data/model_df.csv')

            # 배송지연 여부에 따른 이탈 재주문
            prop_df = (
                model_df.groupby(model_df['delay_days'] > 0)['churn']
                .value_counts(normalize=True)
                .rename("proportion")
                .reset_index()
            )

            # 전처리: 텍스트용 컬럼 추가
            prop_df['배송 지연 여부'] = prop_df['delay_days'].map({False: '정시/조기', True: '지연'})
            prop_df['이탈 여부'] = prop_df['churn'].map({0: '재주문', 1: '이탈'})

            # Altair 시각화
            chart = alt.Chart(prop_df).mark_bar().encode(
                x=alt.X('배송 지연 여부:N', title='배송 상황', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('proportion:Q', title='비율', axis=alt.Axis(format='.0%')),
                color=alt.Color('이탈 여부:N',
                                scale=alt.Scale(range=['#2E86AB', '#A23B72']),
                                legend=alt.Legend(orient='top', titleFontSize=12, labelFontSize=11)),
                xOffset='이탈 여부:N',
                tooltip=['배송 지연 여부:N', '이탈 여부:N',
                         alt.Tooltip('proportion:Q', format='.1%', title='비율')]
            ).properties(
                width='container',
                height=350
            )

            st.altair_chart(chart, use_container_width=True)

        # 구분선 추가
        st.markdown("---")

        # 하단 행: 배송 소요일별 이탈률 (전체 너비 사용)
        st.markdown("##### 📅 배송 소요 기간별 이탈률 추이")

        # 배송 소요일에 따른 이탈 분포
        # 7일 단위로 구간화
        bin_width = 7
        max_day = int(np.ceil(model_df['total_days'].max()))
        bins = np.arange(0, max_day + bin_width, bin_width)
        model_df['days_bin'] = pd.cut(model_df['total_days'], bins=bins, right=False)

        # 각 bin 이탈률(%) 계산
        bin_churn = (
            model_df
            .groupby('days_bin', observed=True)['churn']
            .mean()
            .mul(100)
            .reset_index()
        )
        bin_churn['배송 소요일'] = bin_churn['days_bin'].apply(lambda x: int(x.left + bin_width / 2))

        # 전체 평균 이탈률 계산
        avg_churn_rate = model_df['churn'].mean() * 100

        # Altair 시각화
        bars = alt.Chart(bin_churn).mark_bar(
            color='#F4A580',
            opacity=0.8,
            stroke='#E8956B',
            strokeWidth=1
        ).encode(
            x=alt.X('배송 소요일:O', title='배송 소요일 (7일 단위 중심값)',
                    axis=alt.Axis(labelAngle=0, titleFontSize=12)),
            y=alt.Y('churn:Q', title='이탈률 (%)',
                    axis=alt.Axis(titleFontSize=12, format='.1f')),
            tooltip=[
                alt.Tooltip('배송 소요일:O', title='배송 소요일'),
                alt.Tooltip('churn:Q', title='이탈률 (%)', format='.2f')
            ]
        )

        # 전체 평균 이탈률 선 추가
        mean_line = alt.Chart(pd.DataFrame({'평균 이탈률': [avg_churn_rate]})).mark_rule(
            color='#D32F2F',
            strokeDash=[8, 4],
            strokeWidth=2
        ).encode(
            y=alt.Y('평균 이탈률:Q'),
            tooltip=[alt.Tooltip('평균 이탈률:Q', format='.2f', title='전체 평균 이탈률 (%)')]
        )

        # 평균선 텍스트 라벨
        mean_text = alt.Chart(pd.DataFrame({
            'x': [bin_churn['배송 소요일'].max() * 0.8],
            'y': [avg_churn_rate + 2],
            'text': [f'전체 평균: {avg_churn_rate:.1f}%']
        })).mark_text(
            align='center',
            fontSize=11,
            color='#D32F2F',
            fontWeight='bold'
        ).encode(
            x='x:Q',
            y='y:Q',
            text='text:N'
        )

        final_chart = (bars + mean_line + mean_text).properties(
            width='container',
            height=400,
            title=alt.TitleParams(
                text='배송 소요 기간에 따른 이탈률 변화',
                fontSize=14,
                anchor='start'
            )
        ).resolve_scale(
            y='shared'
        )

        st.altair_chart(final_chart, use_container_width=True)



        # 간단한 텍스트 인사이트
        st.info(
                "📈 **핵심 발견사항:** "
                "배송이 지연될수록 고객 이탈률이 증가하는 경향을 보입니다. "
                "특히 배송 소요일이 길어질수록 이탈률이 평균보다 높아지므로, "
                "배송 시간 단축이 고객 유지에 중요한 요소임을 알 수 있습니다."
            )

    with tab3:
        # 데이터 불러오기
        last_df = pd.read_csv('data/model_df.csv')

        # delay_days 구간 설정
        last_df['delay_group'] = pd.cut(
            last_df['delay_days'],
            bins=[-100, -10, -1, 0, 1, 5, 10, 100],
            labels=['10일 이상 빠름', '1~9일 빠름', '정시 배송', '1일 지연', '2~5일 지연', '6~10일 지연', '10일 이상 지연']
        )

        # RFM_add_score 구간 설정
        last_df['RFM_score_bin'] = pd.qcut(last_df['RFM_add_score'], q=5, labels=False)

        # 페이지 헤더
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px; margin-bottom: 30px;">
            <h1 style="color: white; margin: 0; font-size: 2.5em;">📊 고객 만족도 리뷰분석 대시보드</h1>
            <p style="color: white; margin-top: 10px; font-size: 1.2em;">배송 성과와 고객 충성도 인사이트</p>
        </div>
        """, unsafe_allow_html=True)

        # 핵심 지표 카드
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            avg_score = last_df['review_score'].mean()
            st.metric(
                label="평균 리뷰 점수",
                value=f"{avg_score:.2f}",
                delta=f"{avg_score - 3.0:.2f} vs 기준점"
            )

        with col2:
            on_time_rate = (last_df['delay_days'] <= 0).mean() * 100
            st.metric(
                label="정시 배송률",
                value=f"{on_time_rate:.1f}%",
                delta=f"{on_time_rate - 50:.1f}% vs 평균"
            )

        with col3:
            avg_response_time = last_df['response_time'].mean()
            st.metric(
                label="평균 응답시간",
                value=f"{avg_response_time:.0f}분",
                delta=f"{avg_response_time - 60:.0f}초 vs 목표"
            )

        with col4:
            high_rfm_rate = (last_df['RFM_score_bin'] >= 3).mean() * 100
            st.metric(
                label="고충성 고객 비율",
                value=f"{high_rfm_rate:.1f}%",
                delta=f"{high_rfm_rate - 40:.1f}% vs 평균"
            )

        st.markdown("---")

        # 첫 번째 차트 섹션
        st.markdown("""
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h2 style="color: #2c3e50; margin-bottom: 15px;">🚚 배송 성과별 고객 만족도 분석</h2>
            <p style="color: #7f8c8d;">배송 지연 정도에 따른 리뷰 점수 분포를 확인하세요</p>
        </div>
        """, unsafe_allow_html=True)

        # 차트 옵션
        chart_col1, chart_col2 = st.columns([3, 1])

        with chart_col2:
            st.markdown("#### 차트 설정")
            show_outliers = st.checkbox("이상치 표시", value=True)
            color_palette = st.selectbox(
                "색상 테마",
                ["Set2", "viridis", "husl", "Set3", "pastel"],
                index=0
            )
            chart_style = st.radio(
                "차트 스타일",
                ["박스플롯", "바이올린플롯"],
                index=0
            )

        with chart_col1:
            fig1, ax1 = plt.subplots(figsize=(16, 8))

            if chart_style == "박스플롯":
                box_plot = sns.boxplot(
                    data=last_df,
                    x='delay_group',
                    y='review_score',
                    palette=color_palette,
                    ax=ax1,
                    showfliers=show_outliers
                )
            else:
                box_plot = sns.violinplot(
                    data=last_df,
                    x='delay_group',
                    y='review_score',
                    palette=color_palette,
                    ax=ax1
                )

            ax1.set_xlabel('배송 지연일 그룹', fontsize=14, fontweight='bold', labelpad=15)
            ax1.set_ylabel('리뷰 점수', fontsize=14, fontweight='bold', labelpad=15)
            ax1.set_title('배송 성과별 고객 만족도 분포', fontsize=18, fontweight='bold', pad=20)
            ax1.tick_params(axis='x', rotation=20, labelsize=11)
            ax1.tick_params(axis='y', labelsize=11)
            ax1.grid(True, linestyle='--', alpha=0.3, color='gray')
            ax1.set_facecolor('#fafafa')

            # 평균선 추가
            ax1.axhline(y=last_df['review_score'].mean(), color='red', linestyle='--', alpha=0.7, label='전체 평균')
            ax1.legend()

            plt.tight_layout()
            st.pyplot(fig1)

        # 인사이트 박스
        col1, col2 = st.columns(2)
        with col1:
            early_delivery_score = last_df[last_df['delay_group'].isin(['10일 이상 빠름', '1~9일 빠름'])]['review_score'].mean()
            st.info(f"🚀 **빠른 배송 그룹 평균 점수**: {early_delivery_score:.2f}")

        with col2:
            late_delivery_score = last_df[last_df['delay_group'].isin(['6~10일 지연', '10일 이상 지연'])]['review_score'].mean()
            st.warning(f"⚠️ **지연 배송 그룹 평균 점수**: {late_delivery_score:.2f}")

        st.markdown("---")

        # 두 번째 차트 섹션
        st.markdown("""
        <div style="background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h2 style="color: #2c3e50; margin-bottom: 15px;">💎 고객 충성도별 설문 참여도 분석</h2>
            <p style="color: #7f8c8d;">RFM 점수 구간별 설문 응답 소요시간을 비교해보세요</p>
        </div>
        """, unsafe_allow_html=True)

        # 두 번째 차트 옵션
        chart2_col1, chart2_col2 = st.columns([3, 1])

        with chart2_col2:
            st.markdown("#### 차트 설정")
            show_outliers2 = st.checkbox("이상치 표시", value=True, key="outliers2")
            color_palette2 = st.selectbox(
                "색상 테마",
                ["Set3", "viridis", "husl", "Set2", "pastel"],
                index=0,
                key="palette2"
            )
            show_mean = st.checkbox("평균선 표시", value=True)

        with chart2_col1:
            fig2, ax2 = plt.subplots(figsize=(14, 8))

            box_plot2 = sns.boxplot(
                x='RFM_score_bin',
                y='response_time',
                data=last_df,
                palette=color_palette2,
                ax=ax2,
                showfliers=show_outliers2
            )

            ax2.set_xlabel('RFM 점수 구간 (0: 낮음 ~ 4: 높음)', fontsize=14, fontweight='bold', labelpad=15)
            ax2.set_ylabel('설문응답 소요시간 (초)', fontsize=14, fontweight='bold', labelpad=15)
            ax2.set_title('고객 충성도별 설문 참여도 분석', fontsize=18, fontweight='bold', pad=20)
            ax2.tick_params(axis='both', labelsize=12)
            ax2.grid(True, linestyle='--', alpha=0.3, color='gray')
            ax2.set_facecolor('#fafafa')

            if show_mean:
                ax2.axhline(y=last_df['response_time'].mean(), color='red', linestyle='--', alpha=0.7, label='전체 평균')
                ax2.legend()

            plt.tight_layout()
            st.pyplot(fig2)

        # 통계 정보 표시
        st.markdown("#### 📈 주요 통계 정보")

        stats_col1, stats_col2, stats_col3 = st.columns(3)

        with stats_col1:
            correlation = last_df['RFM_score_bin'].corr(last_df['response_time'])
            st.metric(
                label="RFM-응답시간 ",
                value=f"{correlation:.3f}",
                delta="상관관계 강도"
            )

        with stats_col2:
            high_rfm_avg_time = last_df[last_df['RFM_score_bin'] >= 3]['response_time'].mean()
            st.metric(
                label="고충성 고객 평균 응답시간",
                value=f"{high_rfm_avg_time:.0f}분",
                delta=f"{high_rfm_avg_time - avg_response_time:.0f}1 분 차이"
            )

        with stats_col3:
            low_rfm_avg_time = last_df[last_df['RFM_score_bin'] <= 1]['response_time'].mean()
            st.metric(
                label="저충성 고객 평균 응답시간",
                value=f"{low_rfm_avg_time:.0f}분",
                delta=f"{low_rfm_avg_time - avg_response_time:.0f}1 분 차이"
            )

        # 데이터 테이블 (선택사항)
        with st.expander("📋 상세 데이터 보기"):
            st.markdown("#### 배송 그룹별 통계")
            delay_stats = last_df.groupby('delay_group').agg({
                'review_score': ['mean', 'std', 'count'],
                'response_time': ['mean', 'std']
            }).round(2)
            st.dataframe(delay_stats, use_container_width=True)

            st.markdown("#### RFM 점수별 통계")
            rfm_stats = last_df.groupby('RFM_score_bin').agg({
                'review_score': ['mean', 'std', 'count'],
                'response_time': ['mean', 'std']
            }).round(2)
            st.dataframe(rfm_stats, use_container_width=True)

        # 푸터
        st.markdown("""
        <div style="text-align: center; padding: 20px; margin-top: 40px; 
                    background: #ecf0f1; border-radius: 10px;">
            <p style="color: #7f8c8d; margin: 0;">
                💡 <strong>인사이트</strong>: 배송 성과가 높을수록 고객만족도가 높았습니다 
                RFM분석결과 고충성고객이 설문응답시간이 짧았습니다
            </p>
        </div>
        """, unsafe_allow_html=True)

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

# Footer
st.markdown("---")
html_sticky_footer = """
<style>
    body {
        display: flex;
        flex-direction: column;
        min-height: 100vh;
        margin: 0;
    }
    footer {
        padding: 20px;
        background-color: #f2f2f2);
        width: 100%;
        position: fixed;
        bottom: 0;
        left: 0; /* fix positioned left edge */
        box-sizing: border-box; /* fix padding issues */
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
</style>

<footer>
    <div style="display: flex; align-items: center;">
        <div style="margin-left: 10px;">
            <span style="font-size: 12px; color: #666;">Data Sapiens &copy;2024</span>
        </div>
    </div>
    <div>
        <a href="https://github.com/unayuna" style="text-decoration: none; color: #333; margin: 0 10px;"><img src="https://avatars.githubusercontent.com/u/84783346?v=4" style="width: 30px; height: 30px; border-radius: 50%;"></a>
        <a href="https://github.com/jung33010" style="text-decoration: none; color: #333; margin: 0 10px;"><img src="https://avatars.githubusercontent.com/u/198284688?v=4" style="width: 30px; height: 30px; border-radius: 50%;"></a>
        <a href="https://github.com/YiseoY" style="text-decoration: none; color: #333; margin: 0 10px;"><img src="https://avatars.githubusercontent.com/u/205759568?v=4" style="width: 30px; height: 30px; border-radius: 50%;"></a> 
        <a href="https://github.com/jinijini20" style="text-decoration: none; color: #333; margin: 0 10px;"><img src="https://avatars.githubusercontent.com/u/178241320?v=4" style="width: 30px; height: 30px; border-radius: 50%;"></a>
        <a href="https://github.com/Sumi-Lee" style="text-decoration: none; color: #333; margin: 0 10px;"><img src="https://avatars.githubusercontent.com/u/90227362?v=4" style="width: 30px; height: 30px; border-radius: 50%;"></a>
    </div>
</footer>
"""

# Display the custom sticky footer
st.markdown(html_sticky_footer, unsafe_allow_html=True)