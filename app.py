import os
import gdown
import joblib
import streamlit as st
import numpy as np

MODEL_PATH = "rf_model.joblib"
URL = "https://drive.google.com/uc?id=1oI5bGCNDJZGLyKKosz9Axa1IbEozOw9b"  # direct download link

# Download model if missing or incomplete
if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 100_000_000:
    with st.spinner("Downloading model..."):
        gdown.download(URL, MODEL_PATH, quiet=False, fuzzy=True)

# Load model
try:
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model = None

# Streamlit UI
st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("Fraud Detection / ML Prediction App")

feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

if st.button("Predict"):
    if model:
        X = np.array([[feature1, feature2, feature3]])
        try:
            prediction = model.predict(X)
            st.success(f"Prediction: {prediction[0]}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.error("Model not loaded.")
