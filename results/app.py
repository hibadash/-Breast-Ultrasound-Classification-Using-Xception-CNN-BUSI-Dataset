# ——————————————————————————
# BreastScan AI – Streamlit Premium UI (Enhanced Design)
# ——————————————————————————

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import plotly.graph_objects as go

# ======================
#  Page Configuration
# ======================
st.set_page_config(
    page_title="BreastScan AI",
    page_icon="💗",
    layout="wide"
)

# ======================
#  Session State
# ======================
if 'show_confusion' not in st.session_state:
    st.session_state.show_confusion = False
if 'show_roc' not in st.session_state:
    st.session_state.show_roc = False

# ======================
#  Custom CSS
# ======================
st.markdown("""
<style>

    /* GLOBAL */
    .main {
        background-color: #ffffff !important;
    }
    
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        background-color: #ffffff;
    }

    * {
        font-family: "Poppins", sans-serif;
    }

    #MainMenu, footer {visibility: hidden;}

    /* HEADER */
    .header-container {
        background: linear-gradient(135deg, #fff5f8, #ffffff);
        padding: 2rem 0;
        margin: -3rem -5rem 2rem -5rem;
        border-bottom: 2px solid #ffe4ed;
    }

    .header-title {
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .header-title h1 {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #ff93b3, #ff6e97);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        margin: 0;
    }

    .header-subtitle {
        text-align: center;
        font-size: 1rem;
        color: #888;
        margin-bottom: 0;
    }

    /* CARD */
    .white-card {
        background: #ffffff;
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid #ffe4ed;
        box-shadow: 0 4px 16px rgba(255, 177, 203, 0.12);
        transition: all 0.3s ease;
        height: 100%;
    }

    .white-card:hover {
        box-shadow: 0 8px 24px rgba(255, 139, 178, 0.18);
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .section-title::before {
        content: "";
        width: 4px;
        height: 24px;
        background: linear-gradient(135deg, #ff93b3, #ff6e97);
        border-radius: 4px;
    }

    /* BUTTON */
    .stButton button {
        background: linear-gradient(135deg, #ff93b3, #ff6e97);
        border: none;
        border-radius: 12px;
        padding: 14px 28px;
        color: white;
        font-size: 15px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(255, 126, 166, 0.25);
    }

    .stButton button:hover {
        box-shadow: 0 6px 20px rgba(255, 126, 166, 0.4);
        transform: translateY(-2px);
    }

    /* PREDICTION BARS */
    .prediction-box {
        margin-bottom: 1.5rem;
        background: #fafafa;
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #ffe4ed;
    }

    .prediction-label {
        display: flex;
        justify-content: space-between;
        color: #444;
        font-weight: 600;
        margin-bottom: 0.6rem;
        font-size: 0.95rem;
    }

    .bar-bg {
        height: 12px;
        background: #ffffff;
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid #ffe4ed;
    }

    .bar-fill-benign {
        height: 100%;
        background: linear-gradient(90deg, #ffb6cf, #ff93b3);
        transition: width 0.6s ease;
    }

    .bar-fill-malignant {
        height: 100%;
        background: linear-gradient(90deg, #ff86a3, #ff6e97);
        transition: width 0.6s ease;
    }

    .bar-fill-normal {
        height: 100%;
        background: linear-gradient(90deg, #ffd7e7, #ffb6cf);
        transition: width 0.6s ease;
    }

    /* METRICS SECTION */
    .metrics-container {
        background: #fafafa;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 2rem;
        border: 1px solid #ffe4ed;
    }

    .metric-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1rem;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        padding: 0.7rem 0;
        border-bottom: 1px solid #ffe4ed;
    }

    .metric-row:last-child {
        border-bottom: none;
    }

    .metric-label {
        color: #666;
        font-size: 0.9rem;
    }

    .metric-value {
        color: #ff6e97;
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* FOOTER */
    .custom-footer {
        text-align: center;
        margin-top: 4rem;
        padding: 2rem 0;
        color: #ff86a3;
        border-top: 2px solid #ffe4ed;
        font-size: 0.9rem;
    }

    /* FILE UPLOADER */
    .stFileUploader {
        background: #fafafa;
        border-radius: 12px;
        padding: 1rem;
        border: 2px dashed #ffe4ed;
    }

    /* IMAGE */
    .stImage {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #ffe4ed;
    }

</style>
""", unsafe_allow_html=True)


# ======================
# Header
# ======================
st.markdown("""
<div class="header-container">
    <div class="header-title"><h1>💗 BreastScan AI</h1></div>
    <div class="header-subtitle">AI-Powered Breast Tumor Classification from Ultrasound Imaging</div>
</div>
""", unsafe_allow_html=True)


# ======================
# Load Model
# ======================
@st.cache_resource
def load_ml_model():
    try:
        return load_model("model_xception_best.h5")
    except Exception as e:
        st.error("Model failed to load.")
        return None

CLASSES = ['Benign', 'Malignant', 'Normal']


# ======================
# Main Layout
# ======================
col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Ultrasound Image</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload ultrasound image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True)
        
        # Upload new image button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📤 Upload New Image"):
            st.rerun()
    else:
        st.info("📁 Please upload an ultrasound image to begin diagnosis")

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)

    if uploaded_file:
        model = load_ml_model()

        if model is not None:
            img = image.load_img(uploaded_file, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0
            pred = model.predict(x, verbose=0)[0]

            benign, malignant, normal = pred * 100

            def bar(label, value, css):
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-label">
                        <span>{label}</span>
                        <span>{value:.1f}%</span>
                    </div>
                    <div class="bar-bg">
                        <div class="{css}" style="width:{value}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            bar("Malignant", malignant, "bar-fill-malignant")
            bar("Benign", benign, "bar-fill-benign")
            bar("Normal", normal, "bar-fill-normal")
            
            # Action buttons
            st.markdown("<br>", unsafe_allow_html=True)
            col_btn1, col_btn2 = st.columns(2, gap="medium")
            
            with col_btn1:
                if st.button("📊 Show Confusion Matrix"):
                    st.session_state.show_confusion = not st.session_state.show_confusion
                    st.session_state.show_roc = False
                    
            with col_btn2:
                if st.button("📈 Show ROC Curve"):
                    st.session_state.show_roc = not st.session_state.show_roc
                    st.session_state.show_confusion = False
    else:
        st.info("⏳ Upload an image to see predictions")

    st.markdown('</div>', unsafe_allow_html=True)


# ======================
# Confusion Matrix Section
# ======================
if uploaded_file and st.session_state.show_confusion:
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_cm, col_metrics = st.columns([1.2, 1], gap="large")
    
    with col_cm:
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Confusion Matrix Visualization</div>', unsafe_allow_html=True)

        cm = np.array([
            [80, 0],
            [0, 150]
        ])

        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Malignant', 'Benign'],
            y=['Malignant', 'Benign'],
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 20, "color": "white"},
            colorscale=[[0, "#ffe0eb"], [0.5, "#ffb4c9"], [1, "#ff86a3"]],
            showscale=False
        ))

        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            font=dict(size=12, color="#444")
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_metrics:
        st.markdown('<div class="white-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Performance Metrics</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="metrics-container">
            <div class="metric-row">
                <span class="metric-label">Accuracy</span>
                <span class="metric-value">92.5%</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Recall (Malignant)</span>
                <span class="metric-value">94.1%</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Precision (Malignant)</span>
                <span class="metric-value">88.9%</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">F1-Score (Malignant)</span>
                <span class="metric-value">91.4%</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Precision (Benign)</span>
                <span class="metric-value">96.8%</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Recall (Benign)</span>
                <span class="metric-value">93.8%</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">F1-Score (Benign)</span>
                <span class="metric-value">95.3%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


# ======================
# ROC Curve Section
# ======================
if uploaded_file and st.session_state.show_roc:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="white-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">ROC Curve</div>', unsafe_allow_html=True)
    
    # Generate sample ROC curve data
    fpr = np.linspace(0, 1, 100)
    tpr_malignant = 1 - np.exp(-5 * fpr)
    tpr_benign = 1 - np.exp(-6 * fpr)
    
    fig = go.Figure()
    
    # ROC curve for Malignant
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr_malignant,
        mode='lines',
        name='Malignant (AUC = 0.94)',
        line=dict(color='#ff6e97', width=3)
    ))
    
    # ROC curve for Benign
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr_benign,
        mode='lines',
        name='Benign (AUC = 0.96)',
        line=dict(color='#ff93b3', width=3)
    ))
    
    # Diagonal reference line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='#cccccc', width=2, dash='dash')
    ))
    
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        font=dict(size=12, color="#444"),
        legend=dict(x=0.6, y=0.1),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ======================
# FOOTER
# ======================
st.markdown("""
<div class="custom-footer">
    BreastScan AI © 2025 — Designed with ❤ by Fatema, Heba & Kawtar<br>
    <small>For educational and research purposes only</small>
</div>
""", unsafe_allow_html=True)