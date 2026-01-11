#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import numpy as np
import os
import gdown
import joblib

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "rf_model.joblib"
URL = "https://drive.google.com/uc?id=1oI5bGCNDJZGLyKKosz9Axa1IbEozOw9b"

# Optional: replace with your actual local file size for integrity check
EXPECTED_SIZE = 101449492  # in bytes

# -----------------------------
# Download model if missing
# -----------------------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(URL, MODEL_PATH, quiet=False, fuzzy=True)

# -----------------------------
# Verify model integrity
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found after download.")
    st.stop()

file_size = os.path.getsize(MODEL_PATH)
if EXPECTED_SIZE and file_size != EXPECTED_SIZE:
    st.error(
        f"Model file size mismatch! Expected {EXPECTED_SIZE} bytes, got {file_size} bytes.\n"
        "The model might be corrupted. Please upload the model manually."
    )
    st.stop()

# -----------------------------
# Load model safely
# -----------------------------
try:
    model = joblib.load(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(
        f"Failed to load model: {e}\n"
        "This usually happens if the model file is corrupted or incompatible.\n"
        "Consider re-uploading the model manually."
    )
    model = None

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="ML Model Deployment", layout="centered")
st.title("Fraud Detection / ML Prediction App")

# Example inputs (update based on your actual features)
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

# Prediction button
if st.button("Predict"):
    if model is not None:
        X = np.array([[feature1, feature2, feature3]])
        try:
            prediction = model.predict(X)
            st.success(f"Prediction: {prediction[0]}")
        except Exception as pred_error:
            st.error(f"Prediction failed: {pred_error}")
    else:
        st.error("Model not loaded, cannot make predictions.")
