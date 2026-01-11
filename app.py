#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import numpy as np
import os
import gdown
import joblib

MODEL_PATH = "rf_model.joblib"

if not os.path.exists(MODEL_PATH):
    import gdown
    gdown.download("https://drive.google.com/uc?id=1oI5bGCNDJZGLyKKosz9Axa1IbEozOw9b",
                   MODEL_PATH, quiet=False)

print("File exists:", os.path.exists(MODEL_PATH))
print("File size:", os.path.getsize(MODEL_PATH))

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model = None


st.set_page_config(page_title="ML Model Deployment", layout="centered")

st.title("Fraud Detection / ML Prediction App")

# Example inputs (change based on your features)
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

if st.button("Predict"):
    X = np.array([[feature1, feature2, feature3]])
    prediction = model.predict(X)

    st.success(f"Prediction: {prediction[0]}")

