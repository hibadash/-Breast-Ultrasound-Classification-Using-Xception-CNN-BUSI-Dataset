"""
Module: streamlit_app.py
Description: Interface utilisateur Streamlit pour tester le modèle.
"""

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Breast Ultrasound Classifier", layout="centered")

st.title("🩺 Breast Ultrasound Classifier (Xception)")

uploaded_file = st.file_uploader("Upload an ultrasound image", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    model = tf.keras.models.load_model("app/model/model_xception_best.h5")
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = model.predict(img_array)

    labels = ["Normal", "Benign", "Malignant"]
    st.subheader(f"Prediction: **{labels[np.argmax(pred)]}**")
