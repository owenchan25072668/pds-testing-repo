import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np

# --- 1. SETUP & MODEL LOADING ---
st.set_page_config(page_title="Australia Rain Predictor", layout="wide")

@st.cache_resource
def load_model():
    local_filename = 'rf_model.joblib'
    if not os.path.exists(local_filename):
        # Your specific file ID from the link you provided
        file_id = '1oI5bGCNDJZGLyKKosz9Axa1IbEozOw9b'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, local_filename, quiet=False)
    return joblib.load(local_filename)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("🌧️ Will it Rain Tomorrow in Australia?")
st.markdown("Enter the weather details below to predict rain probability.")

# --- 2. USER INPUTS (SIDEBAR) ---
st.sidebar.header("Select Weather Conditions")

def user_input_features():
    # A. DATE INPUT
    date_pick = st.sidebar.date_input("Date")
    
    # B. CATEGORICAL INPUTS (Lists based on standard AUS weather data)
    locations = ['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree', 
                 'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond', 
                 'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown', 
                 'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat', 
                 'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura', 
                 'Nhill', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns', 
                 'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa', 
                 'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport', 
                 'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston', 
                 'AliceSprings', 'Darwin', 'Katherine', 'Uluru']
    
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

    location = st.sidebar.selectbox('Location', locations)
    wind_gust_dir = st.sidebar.selectbox('Wind Gust Direction', directions)
    wind_dir_9am = st.sidebar.selectbox('Wind Direction 9am', directions)
    wind_dir_3pm = st.sidebar.selectbox('Wind Direction 3pm', directions)
    rain_today = st.sidebar.selectbox('Did it rain today?', ['No', 'Yes'])

    # C. NUMERICAL INPUTS (Sliders with typical ranges)
    min_temp = st.sidebar.slider('Min Temp (°C)', -10.0, 40.0, 15.0)
    max_temp = st.sidebar.slider('Max Temp (°C)', -5.0, 50.0, 25.0)
    rainfall = st.sidebar.number_input('Rainfall (mm)', 0.0, 400.0, 0.0)
    evaporation = st.sidebar.number_input('Evaporation (mm)', 0.0, 100.0, 5.0)
    sunshine = st.sidebar.slider('Sunshine (hours)', 0.0, 15.0, 8.0)
    
    wind_gust_speed = st.sidebar.slider('Wind Gust Speed (km/h)', 0.0, 135.0, 40.0)
    wind_speed_9am = st.sidebar.slider('Wind Speed 9am (km/h)', 0.0, 130.0, 15.0)
    wind_speed_3pm = st.sidebar.slider('Wind Speed 3pm (km/h)', 0.0, 130.0, 20.0)
    
    humidity_9am = st.sidebar.slider('Humidity 9am (%)', 0, 100, 70)
    humidity_3pm = st.sidebar.slider('Humidity 3pm (%)', 0, 100, 50)
    
    pressure_9am = st.sidebar.number_input('Pressure 9am (hPa)', 900.0, 1100.0, 1015.0)
    pressure_3pm = st.sidebar.number_input('Pressure 3pm (hPa)', 900.0, 1100.0, 1012.0)
    
    cloud_9am = st.sidebar.slider('Cloud 9am (oktas)', 0, 9, 4)
    cloud_3pm = st.sidebar.slider('Cloud 3pm (oktas)', 0, 9, 4)
    
    temp_9am = st.sidebar.slider('Temp 9am (°C)', -10.0, 45.0, 20.0)
    temp_3pm = st.sidebar.slider('Temp 3pm (°C)', -10.0, 45.0, 23.0)

    # Store in dictionary
    user_data = {
        'Date': date_pick,
        'Location': location,
        'MinTemp': min_temp,
        'MaxTemp': max_temp,
        'Rainfall': rainfall,
        'Evaporation': evaporation,
        'Sunshine': sunshine,
        'WindGustDir': wind_gust_dir,
        'WindGustSpeed': wind_gust_speed,
        'WindDir9am': wind_dir_9am,
        'WindDir3pm': wind_dir_3pm,
        'WindSpeed9am': wind_speed_9am,
        'WindSpeed3pm': wind_speed_3pm,
        'Humidity9am': humidity_9am,
        'Humidity3pm': humidity_3pm,
        'Pressure9am': pressure_9am,
        'Pressure3pm': pressure_3pm,
        'Cloud9am': cloud_9am,
        'Cloud3pm': cloud_3pm,
        'Temp9am': temp_9am,
        'Temp3pm': temp_3pm,
        'RainToday': rain_today
    }
    return user_data

input_data = user_input_features()

# --- 3. PREPROCESSING (The Tricky Part) ---
# We must manually construct the dataframe to match the model's expected columns.

if st.button('Predict'):
    # 1. Date Engineering
    day = input_data['Date'].day
    month = input_data['Date'].month
    year = input_data['Date'].year

    # 2. Encode RainToday
    rain_today_enc = 1 if input_data['RainToday'] == 'Yes' else 0

    # 3. Create the input dictionary with Numerical cols first
    final_input = {
        'MinTemp': input_data['MinTemp'],
        'MaxTemp': input_data['MaxTemp'],
        'Rainfall': input_data['Rainfall'],
        'Evaporation': input_data['Evaporation'],
        'Sunshine': input_data['Sunshine'],
        'WindGustSpeed': input_data['WindGustSpeed'],
        'WindSpeed9am': input_data['WindSpeed9am'],
        'WindSpeed3pm': input_data['WindSpeed3pm'],
        'Humidity9am': input_data['Humidity9am'],
        'Humidity3pm': input_data['Humidity3pm'],
        'Pressure9am': input_data['Pressure9am'],
        'Pressure3pm': input_data['Pressure3pm'],
        'Cloud9am': input_data['Cloud9am'],
        'Cloud3pm': input_data['Cloud3pm'],
        'Temp9am': input_data['Temp9am'],
        'Temp3pm': input_data['Temp3pm'],
        'RainToday_Yes': rain_today_enc, # Assuming OneHotEncoded as Yes/No
        # 'RainToday_No': 1 - rain_today_enc, # Uncomment if your model uses both
        'Year': year,
        'Month': month,
        'Day': day
        # Note: RISK_MM is excluded as it is a target leakage feature
    }

    # 4. Handle One-Hot Encoding for Categoricals
    # We must set the specific selected column to 1, others to 0
    
    # Helper to add dummy variables
    def add_dummies(prefix, val, options_list):
        for opt in options_list:
            col_name = f"{prefix}_{opt}"
            final_input[col_name] = 1 if val == opt else 0

    # Define the lists (must match training data exactly)
    locations_list = ['Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree', 'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond', 'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown', 'Wollongong', 'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat', 'Bendigo', 'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura', 'Nhill', 'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns', 'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa', 'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport', 'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston', 'AliceSprings', 'Darwin', 'Katherine', 'Uluru']
    directions_list = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

    add_dummies('Location', input_data['Location'], locations_list)
    add_dummies('WindGustDir', input_data['WindGustDir'], directions_list)
    add_dummies('WindDir9am', input_data['WindDir9am'], directions_list)
    add_dummies('WindDir3pm', input_data['WindDir3pm'], directions_list)

    # 5. Convert to DataFrame
    df_predict = pd.DataFrame([final_input])

    # 6. Align Columns
    # IMPORTANT: The model expects columns in a specific order.
    # Get the feature names from the model if possible
    try:
        model_columns = model.feature_names_in_
        # Reindex ensures columns match exactly, filling missing ones with 0
        df_predict = df_predict.reindex(columns=model_columns, fill_value=0)
    except:
        st.warning("Could not auto-align columns. Predicting with constructed features...")

    # --- 4. PREDICTION ---
    prediction = model.predict(df_predict)
    probability = model.predict_proba(df_predict)

    st.subheader('Prediction Result')
    if prediction[0] == 1: # Assuming 1 = Yes
        st.error(f"☔ YES, it will likely rain tomorrow! (Confidence: {probability[0][1]*100:.1f}%)")
    else:
        st.success(f"☀️ NO, it will likely be dry. (Confidence: {probability[0][0]*100:.1f}%)")
        
    # Optional: Debugging - Show what was sent to the model
    with st.expander("See Input Data sent to model"):
        st.write(df_predict)
