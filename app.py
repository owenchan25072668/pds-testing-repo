import streamlit as st
import pandas as pd
import joblib
import gdown
import os

# 1. Load the model (with Google Drive download logic)
@st.cache_resource
def load_model():
    # The file name where we will save the model locally on the server
    local_filename = 'rf_model.joblib'

    # Check if the file already exists. If not, download it.
    if not os.path.exists(local_filename):
        # Your specific Google Drive File ID
        file_id = '1oI5bGCNDJZGLyKKosz9Axa1IbEozOw9b' 
        
        # Construct the download URL
        url = f'https://drive.google.com/uc?id={file_id}'
        
        # Download the file
        gdown.download(url, local_filename, quiet=False)

    # Load the model
    return joblib.load(local_filename)

# Load the model into memory
model = load_model()

# --- The rest of your app UI goes here ---
st.title("My Random Forest Predictor")

# Example input (Replace these with your actual columns!)
input_val = st.number_input("Enter a feature value")

if st.button("Predict"):
    # Create a dataframe with the same column names used in training
    # adjusting 'feature_name' to your actual column name
    input_data = pd.DataFrame([[input_val]], columns=['feature_name'])
    
    prediction = model.predict(input_data)
    st.write(f"Prediction: {prediction[0]}")
