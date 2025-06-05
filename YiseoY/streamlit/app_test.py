import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_extras.let_it_rain import rain
import altair as alt

# Page configuration
st.set_page_config(
    page_title="Olist E-commerce Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load the best model
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/best_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please check the path: models/best_model.pkl")
        return None


model = load_model()

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
st.markdown("""
<div class="main-header">
    <h1>🛒 Olist E-commerce Analytics Dashboard</h1>
    <p>Brazilian E-commerce Customer Satisfaction & Business Intelligence</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Dashboard Navigation")
    dashboard_mode = st.selectbox(
        "Select Dashboard Mode",
        ["Overview", "Prediction", "Analytics", "Performance Metrics"]
    )

    st.markdown("---")
    st.markdown("### 🎯 Model Performance")
    st.metric("Accuracy", "0.881", delta="0.04")
    st.metric("F1 Score", "0.933", delta="0.02")
    st.metric("ROC AUC", "0.794", delta="0.228")

# Main Dashboard Content
if dashboard_mode == "Overview":
    # Key Metrics Row
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        st.metric(
            label="📦 Total Orders",
            value="99,441",
            delta="12.3%"
        )

    with col2:
        st.metric(
            label="💰 Revenue",
            value="R$ 15.4M",
            delta="8.7%"
        )

    with col3:
        st.metric(
            label="😊 Satisfaction Rate",
            value="88.1%",
            delta="4.2%"
        )

    with col4:
        st.metric(
            label="🚚 Avg Delivery",
            value="12.5 days",
            delta="-2.1 days"
        )

    with col5:
        st.metric(
            label="🎯 Accuracy",
            value="0.881",
            delta="0.04"
        )

    with col6:
        st.metric(
            label="📊 F1 Score",
            value="0.933",
            delta="0.02"
        )

    with col7:
        st.metric(
            label="📈 ROC AUC",
            value="0.794",
            delta="0.228"
        )

    st.markdown("---")

    # Charts Row
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Model Prediction Center")

        if model is None:
            st.error("Model not loaded. Cannot make predictions.")
        else:
            # Prediction form in compact layout
            with st.expander("🔧 Hyperparameter Filters", expanded=True):
                col_p1, col_p2 = st.columns(2)

                with col_p1:
                    price = st.number_input('Price (R$)', min_value=0.0, value=50.0, key="overview_price")
                    freight_value = st.number_input('Shipping (R$)', min_value=0.0, value=15.0, key="overview_freight")
                    quantity = st.number_input('Quantity', min_value=1, max_value=5, value=1, key="overview_qty")
                    payment_installments = st.slider('Installments', 1, 24, 1, key="overview_install")

                with col_p2:
                    payment_type = st.selectbox("Payment", ["Credit Card", "Debit Card", "Coupon"],
                                                key="overview_payment")
                    seller_category = st.selectbox('Seller Type',
                                                   ['Verified Seller', 'Successful Seller', 'Unverified Seller'],
                                                   key="overview_seller")
                    seller_score = st.slider('Seller Rating', 0, 10, 7, key="overview_rating")
                    distance = st.slider('Distance (km)', 1, 1000, 100, key="overview_distance")

            # Prediction button and result
            if st.button('🔮 Predict Satisfaction', type='primary', use_container_width=True, key="overview_predict"):
                try:
                    # Process inputs
                    if seller_category == "Verified Seller":
                        category_encoded = [1, 0]
                    elif seller_category == "Successful Seller":
                        category_encoded = [0, 1]
                    else:
                        category_encoded = [0, 0]

                    if payment_type == "Debit Card":
                        payment_encoded = [1, 0]
                    elif payment_type == "Credit Card":
                        payment_encoded = [0, 1]
                    else:
                        payment_encoded = [0, 0]

                    payment_value = (price + freight_value) * quantity
                    wait_encoded = [1, 0, 0, 0]  # Default medium wait time

                    features = [
                        price, freight_value, payment_installments, payment_value,
                        seller_score, 0, distance, 0,  # delay_time and discount set to 0
                        payment_encoded[0], payment_encoded[1],
                        *wait_encoded, *category_encoded
                    ]

                    prediction = model.predict([features])[0]

                    if prediction == 1:
                        st.success("🤩 Customer will be SATISFIED!")
                        st.balloons()
                    else:
                        st.error("😡 Customer will be UNSATISFIED")
                        rain(emoji="😡", font_size=30, falling_speed=2, animation_length="0.5")

                    # Additional metrics
                    st.metric("Prediction Confidence", f"{abs(prediction):.0f}")
                    st.metric("Total Order Value", f"R$ {payment_value:.2f}")

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

    with col2:
        st.subheader("🎯 Customer Satisfaction")
        # Sample satisfaction data
        satisfaction_data = {
            'Rating': [1, 2, 3, 4, 5],
            'Count': [2841, 3151, 8287, 19200, 57328]
        }

        fig = px.bar(
            satisfaction_data,
            x='Rating', y='Count',
            title="Review Score Distribution",
            color='Rating',
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

elif dashboard_mode == "Prediction":
    st.markdown("### 🔮 Customer Satisfaction Prediction")

    if model is None:
        st.error("Model not loaded. Cannot make predictions.")
    else:
        # Prediction Form
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 💳 Order Details")
            price = st.number_input('Product Price (R$)', min_value=0.0, value=50.0)
            freight_value = st.number_input('Shipping Cost (R$)', min_value=0.0, value=15.0)
            discount = st.number_input('Discount Amount (R$)', min_value=0.0, value=0.0)
            quantity = st.number_input('Quantity', min_value=1, max_value=5, value=1)

            st.markdown("#### 💰 Payment Information")
            payment_type = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "Coupon"])
            payment_installments = st.slider('Installments', min_value=1, max_value=24, value=1)

        with col2:
            st.markdown("#### 🏪 Seller Information")
            seller_categories = ['Verified Seller', 'Successful Seller', 'Unverified Seller']
            seller_category = st.selectbox('Seller Type', seller_categories)
            seller_review_score = st.slider('Seller Rating Score', min_value=0, max_value=10, value=7)

            st.markdown("#### 🚚 Delivery Information")
            distance_km = st.slider('Distance (km)', min_value=1, max_value=8736, value=500)

            col_date1, col_date2 = st.columns(2)
            with col_date1:
                order_date = st.date_input('Order Date', value=datetime(2018, 1, 1))
            with col_date2:
                delivery_date = st.date_input('Delivery Date', value=datetime(2018, 1, 15))

        # Process inputs
        if seller_category == "Verified Seller":
            category_encoded = [1, 0]
        elif seller_category == "Successful Seller":
            category_encoded = [0, 1]
        else:
            category_encoded = [0, 0]

        if payment_type == "Debit Card":
            payment_encoded = [1, 0]
        elif payment_type == "Credit Card":
            payment_encoded = [0, 1]
        else:
            payment_encoded = [0, 0]

        customer_wait_days = (delivery_date - order_date).days
        payment_value = ((price + freight_value) * quantity) - discount

        # Encode wait time
        if customer_wait_days <= 8:
            wait_encoded = [0, 0, 0, 0]
        elif customer_wait_days <= 16:
            wait_encoded = [1, 0, 0, 0]
        elif customer_wait_days <= 25:
            wait_encoded = [0, 1, 0, 0]
        elif customer_wait_days <= 40:
            wait_encoded = [0, 0, 1, 0]
        else:
            wait_encoded = [0, 0, 0, 1]

        # Prediction
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button('🔮 Predict Customer Satisfaction', type='primary', use_container_width=True):
                try:
                    features = [
                        price, freight_value, payment_installments, payment_value,
                        seller_review_score, 0, distance_km, discount,  # delay_time set to 0 for now
                        payment_encoded[0], payment_encoded[1],
                        *wait_encoded, *category_encoded
                    ]

                    prediction = model.predict([features])[0]
                    probability = model.predict_proba([features])[0] if hasattr(model, 'predict_proba') else None

                    st.markdown("---")
                    st.markdown("### 📊 Prediction Results")

                    col_result1, col_result2 = st.columns(2)

                    with col_result1:
                        if prediction == 1:
                            st.success("🤩 Customer will be SATISFIED!")
                            st.balloons()
                        else:
                            st.error("😡 Customer will be UNSATISFIED")
                            rain(emoji="😡", font_size=50, falling_speed=3, animation_length="1")

                    with col_result2:
                        if probability is not None:
                            satisfied_prob = probability[1] if len(probability) > 1 else probability[0]
                            st.metric("Satisfaction Probability", f"{satisfied_prob:.2%}")

                        st.metric("Wait Time", f"{customer_wait_days} days")
                        st.metric("Total Value", f"R$ {payment_value:.2f}")

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")

elif dashboard_mode == "Analytics":
    st.markdown("### 📊 Business Analytics")

    tab1, tab2, tab3 = st.tabs(["Order", "Delivery", "Review"])

    with tab1:
        # 1행: 월별 매출 추이
        col1 = st.columns(1)[0]

        with col1:
            st.subheader("월별 매출 추이")

            # 데이터 불러오기
            monthly_stats = pd.read_csv("assets/monthly_stats.csv")
            monthly_stats = monthly_stats.sort_values(['year', 'month'])

            monthly_stats['churn_label'] = monthly_stats['churn'].map({0: '재구매', 1: '이탈'})

            fig_revenue = px.line(
                monthly_stats,
                x='year_month',
                y='total_revenue',
                color='churn_label',
                labels={
                    'year_month': '연도-월',
                    'total_revenue': '매출 (원)',
                    'churn_label': ''
                },
                color_discrete_map={
                    '재구매': '#0066FF',  # 비이탈 고객 - 파란색
                    '이탈': '#FF0000'  # 이탈 고객 - 빨간색
                }
            )

            fig_revenue.update_layout(
                yaxis_title="매출 (원)",
                height=400
            )

            # x축 레이블 회전
            fig_revenue.update_xaxes(tickangle=45)

            st.plotly_chart(fig_revenue, use_container_width=True)

        # 2행: 월별 인당 주문금액 & 주문수 복합 그래프
        col2 = st.columns(1)[0]

        with col2:
            st.subheader("인당 주문금액 & 주문수 추이")

            # 데이터 불러오기
            order_counts_by_month = pd.read_csv("assets/order_counts_by_month.csv")

            # 복합 그래프 생성 (subplot 사용)
            from plotly.subplots import make_subplots

            fig_combo = make_subplots(
                specs=[[{"secondary_y": True}]]
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
                    marker_color='#AED6F1',
                    opacity=0.7
                ),
                secondary_y=False,
            )

            fig_combo.add_trace(
                go.Bar(
                    x=churn_orders['year_month'],
                    y=churn_orders['order_count'],
                    name='이탈 주문수',
                    marker_color='#F1948A',
                    opacity=0.7
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
                    line=dict(color='#0066FF', width=3),
                    marker=dict(size=8)
                ),
                secondary_y=True,
            )

            fig_combo.add_trace(
                go.Scatter(
                    x=churn_stats['year_month'],
                    y=churn_stats['avg_order_value'],
                    mode='lines+markers',
                    name='이탈 AOV',
                    line=dict(color='#FF0000', width=3),
                    marker=dict(size=8)
                ),
                secondary_y=True,
            )

            # y축 레이블 설정
            fig_combo.update_yaxes(title_text="주문 수", secondary_y=False)
            fig_combo.update_yaxes(title_text="인당 주문금액 (원)", secondary_y=True)

            # 레이아웃 설정
            fig_combo.update_layout(
                height=400,
                xaxis_title="연도-월",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            fig_combo.update_xaxes(tickangle=45)

            st.plotly_chart(fig_combo, use_container_width=True)

        # 3행: 카테고리 비율 파이차트 2개 (col3, col4)
        col3, col4 = st.columns(2)

        # col3: churn = 0인 경우 파이차트
        with col3:
            st.subheader("인기  카테고리")

            # 데이터 불러오기
            data = pd.read_csv("assets/order_counts_by_category.csv")

            # churn = 0인 데이터 필터링
            churn_0_data = data[data['churn'] == 0]

            # product_category_name_english별 order_id 고유값 개수 집계
            category_counts_0 = churn_0_data.groupby('product_category_name_english')[
                'order_id'].nunique().reset_index()
            category_counts_0.columns = ['category', 'order_count']

            # 상위 5개 카테고리 추출 (내림차순 정렬)
            top5_categories_0 = category_counts_0.nlargest(5, 'order_count')

            # 비율 계산
            total_orders_0 = top5_categories_0['order_count'].sum()
            top5_categories_0['ratio'] = top5_categories_0['order_count'] / total_orders_0

            # 비율 순으로 정렬 (내림차순)
            top5_categories_0 = top5_categories_0.sort_values('ratio', ascending=False)

            # 파이차트 생성
            fig_0 = px.pie(top5_categories_0,
                           values='ratio',
                           names='category',
                           title="재구매: Top 5 카테고리",
                           color_discrete_sequence=px.colors.qualitative.Set3)
            # 시계방향 배치 설정
            fig_0.update_traces(direction='clockwise', sort=False)

            st.plotly_chart(fig_0, use_container_width=True)

        # col4: churn = 1인 경우 파이차트
        with col4:
            st.subheader(" ")

            # churn = 1인 데이터 필터링
            churn_1_data = data[data['churn'] == 1]

            # product_category_name_english별 order_id 고유값 개수 집계
            category_counts_1 = churn_1_data.groupby('product_category_name_english')[
                'order_id'].nunique().reset_index()
            category_counts_1.columns = ['category', 'order_count']

            # 상위 5개 카테고리 추출 (내림차순 정렬)
            top5_categories_1 = category_counts_1.nlargest(5, 'order_count')

            # 비율 계산
            total_orders_1 = top5_categories_1['order_count'].sum()
            top5_categories_1['ratio'] = top5_categories_1['order_count'] / total_orders_1

            # 비율 순으로 정렬 (내림차순)
            top5_categories_1 = top5_categories_1.sort_values('ratio', ascending=False)

            # 파이차트 생성
            fig_1 = px.pie(top5_categories_1,
                           values='ratio',
                           names='category',
                           title="이탈: Top 5 카테고리",
                           color_discrete_sequence=px.colors.qualitative.Set1)

            # 시계방향 배치 설정
            fig_1.update_traces(direction='clockwise', sort=False)

            st.plotly_chart(fig_1, use_container_width=True)

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

elif dashboard_mode == "Performance Metrics":
    st.markdown("### 📈 Model & Business Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Model Metrics")
        metrics_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            'Score': [0.881, 0.875, 0.892, 0.933, 0.794]
        }
        metrics_df = pd.DataFrame(metrics_data)

        fig = px.bar(metrics_df, x='Metric', y='Score',
                     title="Model Performance Metrics",
                     color='Score', color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📊 Business KPIs")

        # Sample KPI data
        kpi_data = {
            'KPI': ['Customer Retention', 'Order Fulfillment', 'On-time Delivery', 'Return Rate'],
            'Current': [85, 92, 78, 12],
            'Target': [90, 95, 85, 8]
        }
        kpi_df = pd.DataFrame(kpi_data)

        fig = px.bar(kpi_df, x='KPI', y=['Current', 'Target'],
                     title="KPI Performance vs Targets",
                     barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    # Detailed metrics table
    st.markdown("#### 📋 Detailed Performance Report")

    performance_data = {
        'Category': ['Customer Satisfaction', 'Delivery Performance', 'Sales Growth', 'Product Quality'],
        'Current Month': ['88.1%', '78.5%', '12.3%', '4.2/5.0'],
        'Previous Month': ['84.2%', '76.1%', '8.7%', '4.0/5.0'],
        'Change': ['+3.9%', '+2.4%', '+3.6%', '+0.2'],
        'Status': ['✅ Good', '⚠️ Needs Improvement', '✅ Excellent', '✅ Good']
    }

    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True)

# Footer
st.markdown("---")
html_footer = """
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 8px; margin-top: 2rem;">
    <p style="color: #666; font-size: 14px;">
        Data Sapiens © 2024 | Olist E-commerce Analytics Dashboard
    </p>
    <p style="color: #888; font-size: 12px;">
        Built with Streamlit & Best Model | Powered by Brazilian E-commerce Data
    </p>
</div>
"""

st.markdown(html_footer, unsafe_allow_html=True)