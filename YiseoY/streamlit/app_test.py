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