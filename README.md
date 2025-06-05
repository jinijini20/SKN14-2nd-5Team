# SKN14-2nd-5Team
# 📺 Olist 고객 이탈 예측 시스템
## 팀 명 : Team5 🧠
## 👥 팀원소개

## 🖥️ 프로젝트

### 📅 개발 기간
- **2025.06.04 ~ 2025.06.05 (총 2일)**

### 🎬 프로젝트 주제
- 머신러닝을 활용한 Olist E-커머스 이탈예측


## 📌 프로젝트 개요

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
   - GridSearchCV를 통한 최적 파라미터 탐색

5. **모델 평가**
   - 정확도(Accuracy)
   - F1 점수(F1 Score)
   - 분류 보고서(Classification Report)
   - 혼동 행렬(Confusion Matrix)
   - ROC 곡선 및 AUC
   
6. **모델 해석**
   - 특성 중요도(Feature Importance) 분석
   - SHAP 값을 통한 모델 해석
   - 시각화를 통한 결과 비교 분석

---

## 주요 발견 사항
- 사용자의 마지막 로그인 후 경과 일수(Last_Login_days)가 이탈 예측에 가장 중요한 지표로 확인되었습니다.
- RandomForest 모델이 F1 점수 0.82로 가장 높은 성능을 보였으며, 다른 모델들도 안정적인 성능을 보였습니다.
- 범주형 변수들(국가, 선호 장르, 선호 시청 시간대 등)을 Label Encoding하여 모델 성능을 향상시켰습니다.
- SMOTE를 통한 데이터 불균형 해소가 모델의 예측 성능 향상에 기여했습니다.
- 트리 기반 모델(RandomForest, XGBoost, LightGBM)이 로지스틱 회귀보다 약 5-8% 더 좋은 성능을 보였습니다.
- 30일 이상 로그인하지 않은 사용자를 이탈 고객으로 정의한 month_churn 특성이 모델 학습에 유용한 정보를 제공했습니다.

---

## 프로젝트 구조
```
netflix-churn-prediction/
│
├── assets  # 이미지 파일
│
│
│
├── data/
│   ├── netflix_reviews.csv  # 리뷰 데이터셋
│   └── netflix_users.csv    # 최종 분석에 사용된 데이터셋
│
├── notebooks/
│   └── netflix_churn_prediction.ipynb    # 전체 분석 과정이 담긴 노트북
│
├── model/
│   ├── LightGBM_best_model.pkl            # LightGBM 모델 (F1: 0.80)
│   ├── LogisticRegression_best_model.pkl  # 로지스틱 회귀 모델 (F1: 0.77)
│   ├── RandomForest_best_model.pkl        # 랜덤 포레스트 모델 (F1: 0.82)
│   ├── XGBoost_best_model.pkl             # XGBoost 모델 (F1: 0.80)
│   ├── label_encoder.pkl                  
│   ├── model.ipynb                        # 예측 모델
│   ├── model_resurts.csv                  # 모델 예측 결과
│   └── scaler.pkl                          
│
├── styles/
│   ├── netflix_style.css                  # 모델 성능 비교 결과
│   └── styles.css                         
├── utils/
│   ├── __pycache__                        
│   ├── stream_cache.py                     
│   └── stream_utils.py                     
│                       
│
│
├── result/
│   ├── LightGBM - Confusion Matrix.png
│   ├── LightGBM - Feature Importance (Top)
│   ├── LightGBM - ROC Curve.png
│   ├── LogisticRegression - Confusion Matrix.png
│   ├── LogisticRegression - ROC Curve.png
│   ├── RandomForest - Confusion Matrix.png
│   ├── RandomForest - Feature Importance (Top 10).png
│   ├── RandomForest - ROC Curve.png
│   ├── XGBoost - Confusion Matrix.png
│   ├── XGBoost - Feature Importance (Top 10).png
│   └── XGBoost - ROC Curve.png 
│
│    
├── 산출물/
│   ├── XGBoost - Confusion Matrix.png
│   ├── XGBoost - Feature Importance (Top 10).png
│   ├── XGBoost - ROC Curve.png
│   ├── XGBoost_best_model.pkl
│   ├── [OT] 2025_플레이데이터_OT_SK네트웍스 Family AI 캠프 12기.pdf   # ppt 자료 pdf
│   ├── data_preprocessing.csv   # 전처리 데이터
│   ├── label_encoder.pkl
│   └── 모델학습결과.csv    
│   
│
├── README.md/                        
├── app.py/                             
└── requirements.txt                   


```

---

## 결과 및 성능

모델 성능 비교 결과는 다음과 같습니다(model_results.csv 기준):

| 모델 | 정확도(Accuracy) | F1 점수(F1 Score) | 최적 파라미터 |
|------|-----------------|-----------------|--------------|
| RandomForest | 0.8201 | 0.8218 | max_depth=None, max_features='sqrt', min_samples_split=2, n_estimators=100 |
| LightGBM | 0.8018 | 0.8036 | feature_fraction=0.8, learning_rate=0.1, n_estimators=100, num_leaves=31 |
| XGBoost | 0.7950 | 0.7986 | colsample_bytree=0.8, learning_rate=0.1, max_depth=5, n_estimators=100, subsample=0.8 |
| LogisticRegression | 0.7670 | 0.7700 | C=1, penalty='l2', solver='liblinear' |

※ 정확한 수치는 model_results.csv 파일에서 확인 가능합니다.

**성능 결과 분석:**
- RandomForest 모델이 정확도 82.01%, F1 점수 82.18%로 가장 높은 성능을 보였습니다.
- 트리 기반 모델(RandomForest, LightGBM, XGBoost)이 로지스틱 회귀보다 우수한 성능을 보였습니다.
- GridSearchCV를 통한 하이퍼파라미터 최적화가 모델 성능 향상에 크게 기여했습니다.
- 모든 모델이 77% 이상의 정확도를 보이며 안정적인 성능을 나타냈습니다.

**중요 특성 (랭킹 기준):**
1. Last_Login_days (마지막 로그인 이후 경과 일수)
2. Watch_Time_Hours (시청 시간)
3. satisfaction_score (만족도 점수)
4. daily_watch_hours (일일 시청 시간)
5. Age (나이)

**각 모델별 특징:**
- **RandomForest**: 앙상블 학습을 통한 안정적인 성능과 높은 해석 가능성을 제공하여 본 프로젝트에서 최고 성능을 기록
- **LightGBM**: 리프 중심 트리 분할 방식으로 빠른 학습 속도와 메모리 효율성 제공
- **XGBoost**: 경사 부스팅 기반으로 높은 예측 정확도를 보이며, 과적합에 강한 특성을 보임
- **LogisticRegression**: 모델 구조가 단순하여 해석이 용이하나, 복잡한 패턴 학습에는 한계 존재




---

## ✅ 향후 개선 방향

### 1. 모델 성능 개선
- 더 많은 피처 활용  
  → 현재 사용 중인 9개 feature 외에도 고객별 시간 흐름 기반 트렌드, 시즌성 등 추가 피처 도입
- 모델 앙상블 기법 적용  
  → LightGBM, XGBoost, CatBoost 등 다양한 모델을 결합해 정확도 향상
- 하이퍼파라미터 튜닝  
  → GridSearchCV, Optuna 등 활용하여 모델 최적화
- 클래스 불균형 처리  
  → SMOTE, 가중치 조정 등을 통해 불만족 고객의 데이터 불균형 대응

---

### 2. UX 및 UI 개선
- 입력값 자동 완성 기능  
  → 고객 ID를 선택하면 자동으로 관련 입력값이 채워지도록 구현
- 예측 근거 시각화  
  → SHAP, LIME 등을 활용하여 예측의 이유를 시각적으로 설명
- UI 디자인 개선  
  → st.markdown, st.expander, st.progress 등으로 더 깔끔하고 직관적인 사용자 경험 제공

---

### 3. 데이터 파이프라인 자동화
- 데이터 정기 업데이트  
  → DB 또는 외부 파일로부터 실시간/주기적 데이터 수집 및 전처리 자동화
- 모델 버전 관리 시스템 도입  
  → MLflow, DVC 등을 활용하여 모델 성능 추적 및 버전 관리
- MLOps 구조 도입  
  → 예측 API, 모델 학습, 모니터링을 포함한 자동화 파이프라인 구축

---

### 4. 웹앱 기능 확장
- 고객 탐색 대시보드  
  → 고객별 상세 이력, 예측 결과 히스토리 제공
- 이탈 가능성 등급화  
  → 이진 분류(만족/불만족) 외에 리스크 점수나 3단계 등급 예측 제공
- 보고서 다운로드 기능  
  → 예측 결과를 PDF, Excel 등으로 저장 및 공유 가능하게 구현

---

### 5. 비즈니스 연결
- 실시간 CRM 연동  
  → 기업 CRM 시스템과 연동하여 예측 결과를 자동 반영
- 캠페인 자동 추천  
  → 이탈 가능성이 높은 고객에게 맞춤형 혜택, 쿠폰, 상담 연결
- A/B 테스트 실시  
  → 예측 결과 기반 마케팅 대응 vs 무대응의 성과 비교 분석

---





---





## 💭 한줄 회고 
- 이수미 : 
- 이원지희 :
- 전정규 : 여러데이터를 다각도로 자세하게 분석해본게 큰 경험이 된것같습니다 팀원분들 모두 고생많으셨습니다.!
- 송유나 : 
- 윤이서


