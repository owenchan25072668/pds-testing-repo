import streamlit as st
import pandas as pd
import joblib
import numpy as np
import gdown
import os

# --- 1. SETUP & MODEL LOADING ---
st.set_page_config(page_title="Purchase Prediction App", layout="centered")

@st.cache_resource
def load_model():
    local_filename = 'rf_model.joblib'
    
    # Download file from Drive if it doesn't exist locally
    if not os.path.exists(local_filename):
        # Your specific Google Drive File ID (from your link)
        file_id = '1oI5bGCNDJZGLyKKosz9Axa1IbEozOw9b'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, local_filename, quiet=False)
    
    return joblib.load(local_filename)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 2. USER INPUTS (SIDEBAR) ---
st.sidebar.header("User Input Features")

def user_input_features():
    # NUMERICAL INPUTS
    # Adjust min/max/value based on your actual data distribution
    age = st.sidebar.slider('Age', 18, 100, 30)
    estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=10000, max_value=200000, value=50000)
    
    # CATEGORICAL INPUTS
    # These must match the options in your original dataset
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))

    data = {
        'Age': age,
        'EstimatedSalary': estimated_salary,
        'Gender': gender
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- 3. PREPROCESSING ---
# We need to process the user input exactly like the training data
# In your notebook, you likely used pd.get_dummies or LabelEncoder for 'Gender'

# Create a copy to avoid SettingWithCopy warnings
processed_input = input_df.copy()

# OPTION A: If you used Label Encoding (Male=1, Female=0 or similar)
# Check your notebook: If you used LabelEncoder, uncomment below:
# gender_map = {'Male': 1, 'Female': 0} # Verify this mapping!
# processed_input['Gender'] = processed_input['Gender'].map(gender_map)

# OPTION B: If you used One-Hot Encoding (pd.get_dummies)
# This usually creates columns like 'Gender_Male' and 'Gender_Female'
# Since we only have one row, we must manually create these columns
processed_input['Gender_Male'] = 1 if input_df['Gender'][0] == 'Male' else 0
processed_input['Gender_Female'] = 1 if input_df['Gender'][0] == 'Female' else 0

# Drop the original 'Gender' column as models don't read text
processed_input = processed_input.drop(columns=['Gender'])

# Ensure column order matches exactly what the model expects
# This list MUST match X_train.columns from your notebook
# You may need to update this list based on your notebook's "X.columns"
expected_columns = ['Age', 'EstimatedSalary', 'Gender_Female', 'Gender_Male'] 

# Reindex ensures all columns are present and in the correct order
# If a column is missing (e.g., if you dropped one dummy variable), fill with 0
processed_input = processed_input.reindex(columns=expected_columns, fill_value=0)

# --- 4. MAIN PAGE UI ---
st.title("🛍️ Purchase Prediction App")
st.write("This app predicts whether a user will purchase a product based on their demographics.")

# Display user input
st.subheader("User Input:")
st.dataframe(input_df)

# Prediction Button
if st.button("Predict"):
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)

    st.subheader("Prediction Result:")
    
    if prediction[0] == 1:
        st.success("✅ Prediction: Will Purchase")
    else:
        st.warning("❌ Prediction: Will NOT Purchase")

    st.subheader("Prediction Probability:")
    st.write(f"Probability of Purchasing: **{prediction_proba[0][1]:.2f}**")
    st.write(f"Probability of Not Purchasing: **{prediction_proba[0][0]:.2f}**")
