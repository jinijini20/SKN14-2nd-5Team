import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="🛒 Olist 전자상거래 가격 분석",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 메인 타이틀
st.title("🛒 Olist 전자상거래 가격 분석 대시보드")
st.markdown("---")

# 사이드바 - 데이터 업로드 및 설정
st.sidebar.header("📊 데이터 및 설정")

# 샘플 데이터 생성 함수 (실제 데이터가 없을 때 사용)
@st.cache_data
def generate_sample_data():
    """실제 Olist 데이터가 없을 때 샘플 데이터 생성"""
    np.random.seed(42)
    n_samples = 10000

    # 고객 데이터
    customers_df = pd.DataFrame({
        'customer_id': [f'cust_{i}' for i in range(2000)],
        'customer_state': np.random.choice(['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'PE', 'GO', 'CE'], 2000)
    })

    # 주문 데이터
    orders_df = pd.DataFrame({
        'order_id': [f'order_{i}' for i in range(n_samples)],
        'customer_id': np.random.choice(customers_df['customer_id'], n_samples),
        'order_purchase_timestamp': pd.date_range('2017-01-01', '2018-12-31', periods=n_samples),
        'order_delivered_customer_date': pd.NaT
    })

    # 배송 시간 추가
    delivery_days = np.random.normal(15, 5, n_samples)
    delivery_days = np.clip(delivery_days, 3, 60)  # 3~60일 사이
    orders_df['order_delivered_customer_date'] = orders_df['order_purchase_timestamp'] + pd.to_timedelta(delivery_days, unit='D')

    # 상품 카테고리
    categories = ['electronics', 'furniture', 'home_appliances', 'sports_leisure', 'computers',
                 'health_beauty', 'watches_gifts', 'toys', 'fashion_bags', 'auto']

    # 주문 아이템 데이터
    order_items_df = pd.DataFrame({
        'order_id': np.random.choice(orders_df['order_id'], n_samples),
        'product_id': [f'prod_{i}' for i in range(n_samples)],
        'price': np.random.lognormal(3, 1, n_samples),  # 로그정규분포로 가격 생성
        'freight_value': np.random.gamma(2, 8, n_samples)  # 감마분포로 배송비 생성
    })

    # 상품 데이터
    products_df = pd.DataFrame({
        'product_id': order_items_df['product_id'].unique(),
        'product_category_name': np.random.choice(categories, len(order_items_df['product_id'].unique()))
    })

    # 카테고리 번역
    categories_df = pd.DataFrame({
        'product_category_name': categories,
        'product_category_name_english': categories
    })

    # 리뷰 데이터
    order_reviews_df = pd.DataFrame({
        'order_id': order_items_df['order_id'],
        'review_score': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.1, 0.15, 0.3, 0.4])
    })

    return customers_df, orders_df, order_items_df, products_df, categories_df, order_reviews_df

# 데이터 로드 또는 생성
@st.cache_data
def load_data():
    try:
        # 실제 데이터 파일들을 로드 시도
        customers_df = pd.read_csv('data/olist_customers_dataset.csv')
        orders_df = pd.read_csv('data/olist_orders_dataset.csv')
        order_items_df = pd.read_csv('data/olist_order_items_dataset.csv')
        products_df = pd.read_csv('data/olist_products_dataset.csv')
        categories_df = pd.read_csv('data/product_category_name_translation.csv')
        order_reviews_df = pd.read_csv('data/olist_order_reviews_dataset.csv')
        st.sidebar.success("✅ 실제 Olist 데이터를 로드했습니다!")
    except:
        # 파일이 없을 경우 샘플 데이터 생성
        customers_df, orders_df, order_items_df, products_df, categories_df, order_reviews_df = generate_sample_data()
        st.sidebar.info("ℹ️ 샘플 데이터를 사용하고 있습니다.")

    return customers_df, orders_df, order_items_df, products_df, categories_df, order_reviews_df

# 데이터 전처리
@st.cache_data
def preprocess_data(customers_df, orders_df, order_items_df, products_df, categories_df, order_reviews_df):
    # 가격 정보가 포함된 DataFrame 결합
    price_df = (
        order_items_df
        .merge(products_df, on='product_id', how='left')
        .merge(categories_df, on='product_category_name', how='left')
    )

    # 날짜 변환 및 배송 시간 계산
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
    orders_df['order_delivered_customer_date'] = pd.to_datetime(orders_df['order_delivered_customer_date'])
    orders_df['shipping_time'] = (orders_df['order_delivered_customer_date'] - orders_df['order_purchase_timestamp']).dt.days

    # 전체 데이터 병합
    merged = pd.merge(order_items_df, order_reviews_df, on='order_id', how='inner')
    merged = pd.merge(merged, orders_df[['order_id', 'customer_id', 'shipping_time', 'order_purchase_timestamp']], on='order_id', how='left')
    merged = pd.merge(merged, customers_df[['customer_id', 'customer_state']], on='customer_id', how='left')

    # 월별 데이터 추가
    merged['order_month'] = pd.to_datetime(merged['order_purchase_timestamp']).dt.to_period('M')

    return price_df, merged

# 데이터 로드
customers_df, orders_df, order_items_df, products_df, categories_df, order_reviews_df = load_data()
price_df, merged = preprocess_data(customers_df, orders_df, order_items_df, products_df, categories_df, order_reviews_df)

# 사이드바 필터
st.sidebar.subheader("🔍 필터 옵션")
price_range = st.sidebar.slider(
    "가격 범위 (BRL)",
    0, int(price_df['price'].max()),
    (0, 1000),
    step=10
)

# 분석 섹션 선택
analysis_option = st.sidebar.selectbox(
    "분석 섹션 선택",
    ["📈 가격 분포 분석", "🏆 카테고리별 분석", "🚚 가격-배송비 관계", "⭐ 리뷰-가격 관계", "🗺️ 지역별 분석", "📅 시계열 분석", "🤖 배송시간 예측"]
)

# 메인 컨텐츠
if analysis_option == "📈 가격 분포 분석":
    st.header("📈 상품 가격 분포 분석")

    col1, col2 = st.columns(2)

    with col1:
        # 가격 분포 히스토그램
        filtered_price = price_df[(price_df['price'] >= price_range[0]) & (price_df['price'] <= price_range[1])]

        fig = px.histogram(
            filtered_price,
            x='price',
            nbins=50,
            title="상품 가격 분포",
            labels={'price': '가격 (BRL)', 'count': '상품 수'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 기본 통계
        st.subheader("📊 기본 통계")
        stats = filtered_price['price'].describe()

        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.metric("평균 가격", f"{stats['mean']:.2f} BRL")
            st.metric("중앙값", f"{stats['50%']:.2f} BRL")
        with col2_2:
            st.metric("최고가", f"{stats['max']:.2f} BRL")
            st.metric("최저가", f"{stats['min']:.2f} BRL")

elif analysis_option == "🏆 카테고리별 분석":
    st.header("🏆 카테고리별 가격 분석")

    # 카테고리별 평균 가격 TOP 10
    category_price = (
        price_df.groupby('product_category_name_english')['price']
        .agg(['mean', 'count', 'std'])
        .sort_values('mean', ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        category_price,
        x='mean',
        y='product_category_name_english',
        orientation='h',
        title="카테고리별 평균 가격 TOP 10",
        labels={'mean': '평균 가격 (BRL)', 'product_category_name_english': '카테고리'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 카테고리별 상세 통계
    st.subheader("📋 카테고리별 상세 통계")
    st.dataframe(category_price.round(2))

elif analysis_option == "🚚 가격-배송비 관계":
    st.header("🚚 가격과 배송비 관계 분석")

    col1, col2 = st.columns(2)

    with col1:
        # 산점도
        filtered_data = price_df[(price_df['price'] <= 1000) & (price_df['freight_value'] <= 75)]

        fig = px.scatter(
            filtered_data.sample(min(5000, len(filtered_data))),  # 성능을 위해 샘플링
            x='price',
            y='freight_value',
            opacity=0.6,
            title="가격 vs 배송비",
            labels={'price': '가격 (BRL)', 'freight_value': '배송비 (BRL)'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 상관계수
        corr = price_df[['price', 'freight_value']].corr().iloc[0,1]
        st.metric("상관계수", f"{corr:.3f}")

        # 가격 구간별 평균 배송비
        price_df['price_bin'] = pd.cut(price_df['price'], 10)
        avg_freight = price_df.groupby('price_bin')['freight_value'].mean().reset_index()
        avg_freight['price_bin_str'] = avg_freight['price_bin'].astype(str)

        fig2 = px.line(
            avg_freight,
            x='price_bin_str',
            y='freight_value',
            title="가격 구간별 평균 배송비"
        )
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)

elif analysis_option == "⭐ 리뷰-가격 관계":
    st.header("⭐ 리뷰 점수와 가격 관계")

    # 가격대별 평균 리뷰 점수
    merged_clean = merged.dropna(subset=['price', 'review_score'])
    merged_clean['price_bin'] = pd.qcut(merged_clean['price'], 10, duplicates='drop')

    avg_score_by_price = merged_clean.groupby('price_bin')['review_score'].mean().reset_index()
    avg_score_by_price['price_range'] = avg_score_by_price['price_bin'].astype(str)

    fig = px.line(
        avg_score_by_price,
        x='price_range',
        y='review_score',
        title="가격 구간별 평균 리뷰 점수",
        markers=True
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # 리뷰 점수별 가격 박스플롯
    fig2 = px.box(
        merged_clean,
        x='review_score',
        y='price',
        title="리뷰 점수별 가격 분포"
    )
    st.plotly_chart(fig2, use_container_width=True)

elif analysis_option == "🗺️ 지역별 분석":
    st.header("🗺️ 지역별 가격 분석")

    # 지역별 평균 가격
    state_price = merged.groupby('customer_state').agg({
        'price': ['mean', 'count', 'std']
    }).round(2)
    state_price.columns = ['평균_가격', '주문_수', '표준편차']
    state_price = state_price.reset_index().sort_values('평균_가격', ascending=False)

    fig = px.bar(
        state_price,
        x='customer_state',
        y='평균_가격',
        title="지역별 평균 상품 가격",
        labels={'customer_state': '지역 (State)', '평균_가격': '평균 가격 (BRL)'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # 지역별 상세 통계
    st.subheader("📋 지역별 상세 통계")
    st.dataframe(state_price)

elif analysis_option == "📅 시계열 분석":
    st.header("📅 월별 가격 변화")

    # 월별 평균 가격
    month_avg_price = merged.groupby('order_month')['price'].mean().reset_index()
    month_avg_price['order_month_str'] = month_avg_price['order_month'].astype(str)

    fig = px.line(
        month_avg_price,
        x='order_month_str',
        y='price',
        title="월별 평균 상품 가격 변화",
        labels={'order_month_str': '월', 'price': '평균 가격 (BRL)'}
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # 월별 주문량
    month_order_count = merged.groupby('order_month').size().reset_index(name='order_count')
    month_order_count['order_month_str'] = month_order_count['order_month'].astype(str)

    fig2 = px.bar(
        month_order_count,
        x='order_month_str',
        y='order_count',
        title="월별 주문 수",
        labels={'order_month_str': '월', 'order_count': '주문 수'}
    )
    fig2.update_xaxes(tickangle=45)
    st.plotly_chart(fig2, use_container_width=True)

elif analysis_option == "🤖 배송시간 예측":
    st.header("🤖 머신러닝 배송시간 예측")

    # 데이터 준비
    ml_data = merged[['price', 'freight_value', 'shipping_time']].dropna()

    if len(ml_data) > 100:  # 충분한 데이터가 있는 경우
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 예측 입력")
            input_price = st.number_input("상품 가격 (BRL)", min_value=0.0, value=100.0, step=10.0)
            input_freight = st.number_input("배송비 (BRL)", min_value=0.0, value=15.0, step=1.0)

            if st.button("배송시간 예측"):
                # 모델 학습
                X = ml_data[['price', 'freight_value']]
                y = ml_data['shipping_time']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # 예측
                prediction = model.predict([[input_price, input_freight]])[0]

                # 모델 성능
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)

                st.success(f"예상 배송시간: **{prediction:.1f}일**")
                st.info(f"모델 정확도 (MAE): {mae:.2f}일")

        with col2:
            st.subheader("📊 특성 중요도")
            # 특성 중요도 시각화
            X = ml_data[['price', 'freight_value']]
            y = ml_data['shipping_time']
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            importance_df = pd.DataFrame({
                'feature': ['가격', '배송비'],
                'importance': model.feature_importances_
            })

            fig = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                title="배송시간 예측 특성 중요도"
            )
            st.plotly_chart(fig, use_container_width=True)

        # 실제 vs 예측 산점도
        st.subheader("🎯 실제 vs 예측 배송시간")
        X = ml_data[['price', 'freight_value']]
        y = ml_data['shipping_time']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        comparison_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred
        })

        fig = px.scatter(
            comparison_df,
            x='actual',
            y='predicted',
            title="실제 vs 예측 배송시간"
        )
        fig.add_shape(
            type="line",
            x0=comparison_df['actual'].min(),
            y0=comparison_df['actual'].min(),
            x1=comparison_df['actual'].max(),
            y1=comparison_df['actual'].max(),
            line=dict(dash="dash", color="red")
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("머신러닝 모델 학습을 위한 충분한 데이터가 없습니다.")

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🛒 Olist 전자상거래 가격 분석 대시보드 | Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)