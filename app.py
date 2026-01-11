import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Weather Predictor",
    page_icon="🌦️",
    layout="centered" # 'centered' looks better for forms than 'wide'
)

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_model():
    local_filename = 'rf_model.joblib'
    if not os.path.exists(local_filename):
        # Your File ID
        file_id = '1oI5bGCNDJZGLyKKosz9Axa1IbEozOw9b'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, local_filename, quiet=False)
    return joblib.load(local_filename)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 3. UI: TITLE & HEADER ---
st.title("🌦️ Australia Rain Predictor")
st.markdown("Adjust the parameters below to predict tomorrow's weather.")

# --- 4. UI: INPUT FORM (Organized by Tabs) ---
# We use a form so the app doesn't reload with every slider movement
with st.form("prediction_form"):
    
    # Create tabs to hide the complexity
    tab1, tab2, tab3 = st.tabs(["📍 Location & Date", "🌡️ Temperature", "💨 Wind & Humidity"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            date_pick = st.date_input("Select Date")
            location = st.selectbox('City / Location', 
                                  ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Canberra', 'Hobart', 'Darwin', 'AliceSprings', 'Cobar', 'NorfolkIsland'])
        with col2:
            rain_today = st.radio("Did it rain today?", ['No', 'Yes'], horizontal=True)

    with tab2:
        col3, col4 = st.columns(2)
        with col3:
            min_temp = st.slider('Min Temp (°C)', -10.0, 40.0, 15.0)
            max_temp = st.slider('Max Temp (°C)', -5.0, 50.0, 25.0)
            sunshine = st.slider('Sunshine (hours)', 0.0, 15.0, 8.0)
        with col4:
            evaporation = st.number_input('Evaporation (mm)', 0.0, 100.0, 5.0)
            rainfall = st.number_input('Rainfall (mm)', 0.0, 400.0, 0.0)

    with tab3:
        col5, col6 = st.columns(2)
        with col5:
            wind_gust_speed = st.slider('Wind Gust Speed (km/h)', 0.0, 135.0, 40.0)
            humidity_3pm = st.slider('Humidity 3pm (%)', 0, 100, 50)
            pressure_3pm = st.number_input('Pressure 3pm (hPa)', 900.0, 1100.0, 1012.0)
        with col6:
            wind_dir = st.selectbox('Wind Direction', ['N', 'S', 'E', 'W', 'NW', 'SE', 'SW', 'NE'])
            cloud_3pm = st.slider('Cloud Cover 3pm (0-9)', 0, 9, 4)

    # A Big "Predict" Button
    submit_btn = st.form_submit_button("🔮 Predict Result", type="primary")

# --- 5. PREDICTION LOGIC ---
if submit_btn:
    # A. Preprocessing (Convert inputs to model format)
    day, month, year = date_pick.day, date_pick.month, date_pick.year
    rain_today_enc = 1 if rain_today == 'Yes' else 0

    # Base Input Dictionary (Numerical)
    # Note: We fill in defaults for fields we didn't ask the user for (to keep UI clean)
    input_data = {
        'MinTemp': min_temp, 'MaxTemp': max_temp, 'Rainfall': rainfall,
        'Evaporation': evaporation, 'Sunshine': sunshine,
        'WindGustSpeed': wind_gust_speed, 'WindSpeed9am': 15.0, 'WindSpeed3pm': 20.0,
        'Humidity9am': 70, 'Humidity3pm': humidity_3pm,
        'Pressure9am': 1015.0, 'Pressure3pm': pressure_3pm,
        'Cloud9am': 4, 'Cloud3pm': cloud_3pm,
        'Temp9am': 20.0, 'Temp3pm': 23.0,
        'RainToday_Yes': rain_today_enc, 'Year': year, 'Month': month, 'Day': day
    }

    # B. Align with Model Columns (Crucial Step)
    # Get features from model
    try:
        model_cols = model.feature_names_in_
    except:
        st.error("Model structure issue. Please check .joblib file.")
        st.stop()

    # Create DataFrame with 0s for all columns
    df_predict = pd.DataFrame(0, index=[0], columns=model_cols)

    # Fill numerical data
    for col, val in input_data.items():
        if col in df_predict.columns:
            df_predict[col] = val

    # Fill One-Hot Encoded data (Location/Wind)
    if f"Location_{location}" in df_predict.columns:
        df_predict[f"Location_{location}"] = 1
    if f"WindGustDir_{wind_dir}" in df_predict.columns:
        df_predict[f"WindGustDir_{wind_dir}"] = 1

    # C. Prediction
    prediction = model.predict(df_predict)[0]
    probs = model.predict_proba(df_predict)[0]
    prob_rain = probs[1]

    # --- 6. DISPLAY RESULTS ---
    st.markdown("---")
    st.subheader("Prediction Report")

    # Layout for results
    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        if prediction == 1:
            st.error("☔ **RAIN**")
            st.write("Prepare your umbrella!")
        else:
            st.success("☀️ **NO RAIN**")
            st.write("Enjoy the sunshine!")

    with res_col2:
        # Simple native Streamlit progress bar
        st.write(f"Confidence Level: **{prob_rain*100:.1f}%**")
        st.progress(prob_rain, text="Chance of Rain")
        
        # Additional metrics
        m1, m2 = st.columns(2)
        m1.metric("Humidity", f"{humidity_3pm}%")
        m2.metric("Pressure", f"{pressure_3pm} hPa")
