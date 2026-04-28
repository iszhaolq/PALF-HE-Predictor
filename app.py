import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

st.set_page_config(
    page_title="Pediatric ALD-HE Predictor",
    page_icon="⚕️",
    layout="wide",  
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* 侧边栏背景色 */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
    /* 预测按钮样式：深蓝绿色 */
    div.stButton > button:first-child {
        background-color: #206a5d;
        color: white;
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
    div.stButton > button:first-child:hover {
        background-color: #144d4c;
        border-color: #144d4c;
    }
    /* 标题颜色 */
    h1 {
        color: #1f4287;
        font-weight: 700;
    }
    /* 结果显示区的样式 */
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_trained_model():
    model_path = "palf_he_rf_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


model = load_trained_model()

if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("📋 Patient Measurements")
    st.info("Enter clinical extreme values captured within the first 5 days.")

    inr = st.number_input("Peak INR", min_value=0.5, max_value=20.0, value=1.2, step=0.1)
    k = st.number_input("Trough Potassium (mmol/L)", min_value=1.0, max_value=10.0, value=4.0, step=0.1)
    ua = st.number_input("Peak Uric Acid (μmol/L)", min_value=10.0, max_value=2000.0, value=200.0, step=10.0)
    crp = st.number_input("Peak CRP (mg/L)", min_value=0.0, max_value=500.0, value=10.0, step=1.0)
    tbil = st.number_input("Peak TBIL (μmol/L)", min_value=1.0, max_value=2000.0, value=20.0, step=5.0)
    alt = st.number_input("Peak ALT (U/L)", min_value=5.0, max_value=20000.0, value=40.0, step=10.0)

    predict_btn = st.button("Predict Risk")

st.title("Pediatric ALD-HE Risk Predictor")
st.markdown("Developed for early identification of Hepatic Encephalopathy (HE) in pediatric Acute Liver Dysfunction.")

if model is None:
    st.error("⚠️ Model file `palf_he_rf_model.pkl` not found. Please ensure it is in the same directory.")
else:

    if predict_btn:

        input_df = pd.DataFrame({
            'INR_最高值': [inr],
            '钾离子_最低值': [k],
            '尿酸_最高值': [ua],
            'CRP_最高值': [crp],
            '总胆红素_最高值': [tbil],
            '谷丙转氨酶_最高值': [alt]
        })


        prob = model.predict_proba(input_df)[0][1]

 
        new_record = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "INR": inr,
            "K (min)": k,
            "UA (max)": ua,
            "CRP": crp,
            "TBIL": tbil,
            "ALT": alt,
            "Probability (%)": round(prob * 100, 2)
        }
        st.session_state.history.insert(0, new_record)  

        st.subheader("📊 Prediction Result")

        if prob < 0.35:
            risk_color = "#2d6a4f"  # 深绿色
            risk_label = "Low Risk"
            advice = "Recommendation: Routine monitoring; prioritize coagulation and electrolyte balance."
        elif prob < 0.70:
            risk_color = "#f39c12"  # 橙色
            risk_label = "Moderate Risk"
            advice = "Recommendation: High vigilance; consider hyperammonemia prevention and increased rounds."
        else:
            risk_color = "#c0392b"  # 深红色
            risk_label = "High Risk"
            advice = "Recommendation: Immediate medical intervention; evaluate ALSS or liver transplant."

        st.markdown(f"""
            <div style="background-color: {risk_color}; padding: 30px; border-radius: 15px; text-align: center;">
                <h1 style="color: white; margin: 0;">{prob * 100:.2f}%</h1>
                <h3 style="color: white; margin: 0;">Predicted Risk: {risk_label}</h3>
            </div>
            """, unsafe_allow_html=True)

        st.info(advice)


st.markdown("---")
st.subheader("🕒 Prediction History")
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.table(history_df)
else:
    st.text("No history records yet.")
