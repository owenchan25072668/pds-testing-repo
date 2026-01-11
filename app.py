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
URL = "https://drive.google.com/uc?id=1oI5bGCNDJZGLyKKosz9Axa1IbEozOw9b"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(URL, MODEL_PATH, quiet=False, fuzzy=True)

model = joblib.load("MODEL_PATH")

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

