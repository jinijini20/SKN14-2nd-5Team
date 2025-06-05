import alt
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
        ["Overview", "Prediction", "Analytics"]
    )

# Main Dashboard Content
if dashboard_mode == "Prediction":
    # Key Metrics Row
    col5, col6, col7 = st.columns(3)

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
    col1, col2, col3, col4 = st.columns(4)

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
    st.markdown("### 📊 Business Analytics")

    tab1, tab2, tab3 = st.tabs(["Order", "Delivery", "Review"])
    # Main Dashboard Content
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Sample sales data
            months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
            sales = [45000, 52000, 48000, 61000, 55000, 67000]

            fig = px.bar(x=months, y=sales, title="Monthly Sales Revenue (R$)")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Sample category data
            categories = ['Electronics', 'Fashion', 'Home', 'Sports', 'Books']
            values = [25, 35, 20, 15, 5]

            fig = px.pie(values=values, names=categories, title="Sales by Category")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("#### 📦 Delivery Distribution")

        col1 = st.columns(1)[0]
        with col1:
            # csv 파일 읽기
            order_churn = pd.read_csv('data/order_churn.csv')

            # 이탈 여부에 따른 주문상태 count 집계
            chart_data = order_churn.groupby(['order_status', 'churn_str']).size().reset_index(name='count')
            sort = ['shipped', 'canceled', 'unavailable', 'processing', 'created', 'approved']

            # Altair 그래프 생성
            chart = alt.Chart(chart_data).mark_bar().encode(
                x=alt.X('order_status:N', title='배송 상태', sort=sort, axis=alt.Axis(labelAngle=0)),
                y=alt.Y('count:Q', title='count'),
                color=alt.Color('churn_str:N', title='이탈 여부', scale=alt.Scale(scheme='redblue'), legend=alt.Legend(title=None)),
                xOffset='churn_str:N', # legend 표시
                tooltip=['order_status', 'churn_str', 'count']
            ).properties(
                width=90,
                height=400,
                title='주문 상태별 이탈 분포'
            )

            # Streamlit에 출력
            st.altair_chart(chart, use_container_width=True)

        col2 = st.columns(1)[0]
        with col2:
            # 배송 지연일에 따른 이탈 분포
            # csv 파일 읽기
            model_df = pd.read_csv('data/model_df.csv')

            # 배송지연 여부에 따른 이탈 재주문
            # 배송지연 (배송완료일 - 예상일)
            # delay_days > 0: 지연 배송
            # delay_days <= 0: 조기 배송
            prop_df = (
                model_df.groupby(model_df['delay_days'] > 0)['churn']
                .value_counts(normalize=True)
                .rename("proportion")
                .reset_index()
            )

            # 전처리: 텍스트용 컬럼 추가
            prop_df['배송 지연 여부'] = prop_df['delay_days'].map({False: '정상', True: '지연'})
            prop_df['이탈 여부'] = prop_df['churn'].map({0: '재주문', 1: '이탈'})

            # Altair 시각화
            chart = alt.Chart(prop_df).mark_bar().encode(
                x=alt.X('배송 지연 여부:N', title='배송 지연 여부', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('proportion:Q', title='이탈률'),
                color=alt.Color('이탈 여부:N', scale=alt.Scale(scheme='redblue'), legend=alt.Legend(title=None)),
                xOffset='이탈 여부:N',
                tooltip=['배송 지연 여부', '이탈 여부', alt.Tooltip('proportion', format='.2%')]
            ).properties(
                width=300,
                height=400,
                title='배송 지연 여부에 따른 이탈률 분포'
            )

            # 출력
            st.altair_chart(chart, use_container_width=True)

        col3 = st.columns(1)[0]
        with col3:
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

            # Altair 시각화
            chart = alt.Chart(bin_churn).mark_bar(color='orange').encode(
                x=alt.X('배송 소요일:O', title='배송 소요일(일) - 7일 단위', axis=alt.Axis(labelAngle=0)),
                y=alt.Y('churn:Q', title='이탈률 (%)'),
                color=alt.value('#F4A580'),
                tooltip=[
                    alt.Tooltip('배송 소요일:O', title='배송 소요일(중심값)'),
                    alt.Tooltip('churn:Q', title='이탈률 (%)', format='.2f')
                ]
            ).properties(
                width=700,
                height=400,
                title='배송 소요 기간에 따른 이탈률 변화'
            )

            # 전체 평균 이탈률 선 추가
            mean_line = alt.Chart(pd.DataFrame({'y': [model_df['churn'].mean() * 100]})).mark_rule(
                color='red', strokeDash=[5, 5]
            ).encode(y='y:Q')

            # Step 6: Streamlit에 출력
            st.altair_chart(chart + mean_line, use_container_width=True)





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