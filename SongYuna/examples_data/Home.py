import base64
import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
import seaborn as sns
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# with open("deployment/style.css", "r", encoding="utf-8") as pred:
# footer_html = f"""{pred.read()}"""
# st.markdown(footer_html, unsafe_allow_html=True)

st.set_page_config(
    page_title="team data",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# st.sidebar.image("deployment/assets/datasapienslogo.png")
st.image("images/olist.png", width=200, use_column_width="never")

st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="images/olist.png" alt="Olist Logo" width="200"/>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """

## Project Overview:
- The "Data Sapiens" dashboard showcases insights derived from a publicly available dataset of e-commerce orders from Olist, comprising 100,000 orders between 2016 and 2018.
- Leveraging machine learning models, namely Catboost and Logistic Regression, it provides valuable information about various aspects of the customer journey, including order status, checkout, and customer reviews.
- The dataset encompasses details about sellers listing products on Olist, along with customer behavior and demographic data.

---

### Models Utilized:
**User Prediction:**
   - Model: XGBoost
   - Description: Predicts customer review scores based on various factors.

---

### Model Evaluation:
- **Churn Prediction:**
  - Evaluation Metrics: Accuracy, ROC Auc ,s F1 Score
  - Performance: Achieved an accuracy of 0.881%, ROC Auc score of 0.794 and F1 score of 0.933%.

---
""", unsafe_allow_html=True)

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