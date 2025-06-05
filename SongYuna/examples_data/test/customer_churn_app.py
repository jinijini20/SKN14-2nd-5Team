import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="고객 이탈 예측 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .churn-risk {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef5350;
    }
    .retain-customer {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #66bb6a;
    }
</style>
""", unsafe_allow_html=True)

# 제목
st.markdown('<h1 class="main-header">🎯 고객 이탈 예측 시스템</h1>', unsafe_allow_html=True)

# 사이드바 - 모델 설정
st.sidebar.header("🔧 모델 설정")

# 더미 모델 생성 (실제 환경에서는 저장된 모델을 로드)
@st.cache_resource
def load_model():
    # 실제 프로젝트에서는 pickle로 저장된 모델을 로드
    model = LogisticRegression(
        max_iter=500,
        class_weight='balanced',
        C=10,
        penalty='l1',
        solver='liblinear',
        random_state=42
    )
    
    scaler = StandardScaler()
    
    # 더미 데이터로 학습 (실제로는 저장된 모델 사용)
    np.random.seed(42)
    X_dummy = np.random.randn(1000, 6)
    y_dummy = np.random.choice([0, 1], 1000, p=[0.7, 0.3])
    
    model.fit(X_dummy, y_dummy)
    scaler.fit(X_dummy)
    
    return model, scaler

model, scaler = load_model()

# 특성 정의
features = ['Recency', 'Frequency', 'Monetary', 'delay_days', 'total_days', 'approval_days']
feature_descriptions = {
    'Recency': '최근 주문일로부터 경과일수',
    'Frequency': '총 주문 횟수',
    'Monetary': '총 구매 금액 (BRL)',
    'delay_days': '배송 지연일 (음수면 조기배송)',
    'total_days': '주문~배송완료 총 소요일',
    'approval_days': '주문~결제승인 소요일'
}

# 메인 레이아웃
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📋 고객 정보 입력")
    
    # 사용자 입력
    user_inputs = {}
    
    with st.expander("🔍 RFM 정보", expanded=True):
        user_inputs['Recency'] = st.slider(
            "Recency (최근 주문 경과일)", 
            min_value=1, max_value=500, value=30,
            help=feature_descriptions['Recency']
        )
        
        user_inputs['Frequency'] = st.slider(
            "Frequency (총 주문 횟수)", 
            min_value=1, max_value=20, value=2,
            help=feature_descriptions['Frequency']
        )
        
        user_inputs['Monetary'] = st.slider(
            "Monetary (총 구매 금액)", 
            min_value=10.0, max_value=10000.0, value=150.0, step=10.0,
            help=feature_descriptions['Monetary']
        )
    
    with st.expander("🚚 배송 정보", expanded=True):
        user_inputs['delay_days'] = st.slider(
            "배송 지연일", 
            min_value=-30, max_value=100, value=0,
            help=feature_descriptions['delay_days']
        )
        
        user_inputs['total_days'] = st.slider(
            "총 소요일", 
            min_value=1, max_value=200, value=15,
            help=feature_descriptions['total_days']
        )
        
        user_inputs['approval_days'] = st.slider(
            "결제 승인 소요일", 
            min_value=0, max_value=10, value=1,
            help=feature_descriptions['approval_days']
        )

with col2:
    st.header("🎯 예측 결과")
    
    # 예측 수행
    input_data = np.array([list(user_inputs.values())])
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    # 결과 표시
    if prediction == 1:
        st.markdown(
            '<div class="prediction-result churn-risk">⚠️ 이탈 위험 고객</div>',
            unsafe_allow_html=True
        )
        risk_level = "높음"
        risk_color = "#ff4444"
    else:
        st.markdown(
            '<div class="prediction-result retain-customer">✅ 유지 고객</div>',
            unsafe_allow_html=True
        )
        risk_level = "낮음"
        risk_color = "#44ff44"
    
    # 확률 표시
    col2_1, col2_2 = st.columns(2)
    with col2_1:
        st.metric(
            "이탈 확률",
            f"{probability[1]:.1%}",
            delta=f"{probability[1]-0.5:.1%}" if probability[1] > 0.5 else f"{0.5-probability[1]:.1%}"
        )
    with col2_2:
        st.metric(
            "유지 확률",
            f"{probability[0]:.1%}",
            delta=f"{probability[0]-0.5:.1%}" if probability[0] > 0.5 else f"{0.5-probability[0]:.1%}"
        )

# 시각화 섹션
st.header("📊 데이터 분석 및 시각화")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 입력값 분석", "📈 특성 중요도", "🔄 RFM 분석", "📦 배송 분석"])

with tab1:
    st.subheader("현재 입력값 분석")
    
    # 입력값을 데이터프레임으로 변환
    input_df = pd.DataFrame([user_inputs])
    
    # 레이더 차트
    fig_radar = go.Figure()
    
    # 정규화된 값 (0-1 스케일)
    normalized_values = []
    for feature in features:
        if feature == 'Recency':
            norm_val = max(0, 1 - user_inputs[feature] / 500)  # Recency는 낮을수록 좋음
        elif feature == 'delay_days':
            norm_val = max(0, 1 - abs(user_inputs[feature]) / 100)  # 지연이 적을수록 좋음
        else:
            # 다른 특성들은 적절한 범위로 정규화
            ranges = {'Frequency': 20, 'Monetary': 10000, 'total_days': 200, 'approval_days': 10}
            norm_val = min(1, user_inputs[feature] / ranges[feature])
        normalized_values.append(norm_val)
    
    fig_radar.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=features,
        fill='toself',
        name='현재 고객',
        line_color=risk_color
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="고객 특성 레이더 차트"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

with tab2:
    st.subheader("특성 중요도 분석")
    
    # 모델 계수 (특성 중요도)
    coefficients = model.coef_[0]
    importance_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=True)
    
    # 수평 막대 차트
    fig_importance = px.bar(
        importance_df, 
        x='Coefficient', 
        y='Feature',
        orientation='h',
        color='Coefficient',
        color_continuous_scale='RdYlBu_r',
        title="특성 중요도 (로지스틱 회귀 계수)"
    )
    fig_importance.update_layout(height=400)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # 특성별 영향도 설명
    st.subheader("특성별 영향도 해석")
    for i, (feature, coef) in enumerate(zip(features, coefficients)):
        impact = "이탈 위험 증가" if coef > 0 else "이탈 위험 감소"
        color = "🔴" if coef > 0 else "🟢"
        st.write(f"{color} **{feature}**: {impact} (계수: {coef:.3f})")

with tab3:
    st.subheader("RFM 분석")
    
    # RFM 히스토그램
    fig_rfm = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Recency', 'Frequency', 'Monetary')
    )
    
    # 샘플 데이터 생성 (실제로는 전체 고객 데이터 사용)
    np.random.seed(42)
    sample_recency = np.random.exponential(50, 1000)
    sample_frequency = np.random.poisson(3, 1000) + 1
    sample_monetary = np.random.lognormal(5, 1, 1000)
    
    fig_rfm.add_trace(
        go.Histogram(x=sample_recency, name="전체 고객", opacity=0.7),
        row=1, col=1
    )
    fig_rfm.add_vline(x=user_inputs['Recency'], line_dash="dash", 
                     line_color="red", row=1, col=1)
    
    fig_rfm.add_trace(
        go.Histogram(x=sample_frequency, name="전체 고객", opacity=0.7),
        row=1, col=2
    )
    fig_rfm.add_vline(x=user_inputs['Frequency'], line_dash="dash", 
                     line_color="red", row=1, col=2)
    
    fig_rfm.add_trace(
        go.Histogram(x=sample_monetary, name="전체 고객", opacity=0.7),
        row=1, col=3
    )
    fig_rfm.add_vline(x=user_inputs['Monetary'], line_dash="dash", 
                     line_color="red", row=1, col=3)
    
    fig_rfm.update_layout(height=400, title_text="RFM 분포 (빨간 선: 현재 고객)")
    st.plotly_chart(fig_rfm, use_container_width=True)

with tab4:
    st.subheader("배송 성과 분석")
    
    # 배송 관련 지표
    col4_1, col4_2, col4_3 = st.columns(3)
    
    with col4_1:
        delay_status = "지연" if user_inputs['delay_days'] > 0 else "정시/조기"
        delay_color = "red" if user_inputs['delay_days'] > 0 else "green"
        st.metric(
            "배송 상태", 
            delay_status,
            delta=f"{user_inputs['delay_days']}일"
        )
    
    with col4_2:
        total_rating = "빠름" if user_inputs['total_days'] <= 10 else "보통" if user_inputs['total_days'] <= 20 else "느림"
        st.metric("총 소요시간", total_rating, delta=f"{user_inputs['total_days']}일")
    
    with col4_3:
        approval_rating = "즉시" if user_inputs['approval_days'] <= 1 else "지연"
        st.metric("결제 승인", approval_rating, delta=f"{user_inputs['approval_days']}일")
    
    # 배송 성과 점수 계산
    delivery_score = 100
    if user_inputs['delay_days'] > 0:
        delivery_score -= user_inputs['delay_days'] * 2
    if user_inputs['total_days'] > 15:
        delivery_score -= (user_inputs['total_days'] - 15) * 1
    if user_inputs['approval_days'] > 1:
        delivery_score -= (user_inputs['approval_days'] - 1) * 5
    
    delivery_score = max(0, delivery_score)
    
    # 게이지 차트
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = delivery_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "배송 성과 점수"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig_gauge.update_layout(height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

# 사이드바 - 추가 정보
st.sidebar.markdown("---")
st.sidebar.subheader("📖 모델 정보")
st.sidebar.info("""
**모델**: Logistic Regression
**특성**: RFM + 배송 관련 6개 변수
**이탈 기준**: R_score ≤ 2 AND F_score ≤ 2
""")

st.sidebar.subheader("💡 개선 제안")
if prediction == 1:
    st.sidebar.warning("""
    **이탈 위험 고객 대응 방안:**
    - 개인화된 할인 쿠폰 제공
    - 배송 서비스 개선
    - 고객 만족도 조사 실시
    - 프리미엄 멤버십 제안
    """)
else:
    st.sidebar.success("""
    **유지 고객 관리 방안:**
    - 로열티 프로그램 참여 유도
    - 신제품 우선 안내
    - 추천 상품 제안
    - 정기적인 만족도 확인
    """)

# 푸터
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>고객 이탈 예측 시스템 v1.0 | Built with Streamlit</div>",
    unsafe_allow_html=True
)