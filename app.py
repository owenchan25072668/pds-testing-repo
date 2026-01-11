import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import requests
import plotly.graph_objects as go
from streamlit_lottie import st_lottie

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="WeatherAI Predictor", page_icon="🌦️", layout="wide")

# --- 2. HELPER FUNCTIONS ---

# Function to load Lottie Animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load assets (Sunny and Rainy animations)
lottie_rain = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_b88nh30c.json")
lottie_sun = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_jq7pxw33.json")

# Coordinates for major locations (to show on map)
# You can add more, these are defaults
city_coords = {
    'Sydney': [-33.8688, 151.2093],
    'Melbourne': [-37.8136, 144.9631],
    'Brisbane': [-27.4705, 153.0260],
    'Perth': [-31.9505, 115.8605],
    'Adelaide': [-34.9285, 138.6007],
    'Canberra': [-35.2809, 149.1300],
    'Hobart': [-42.8821, 147.3272],
    'Darwin': [-12.4634, 130.8456]
}

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    local_filename = 'rf_model.joblib'
    if not os.path.exists(local_filename):
        file_id = '1oI5bGCNDJZGLyKKosz9Axa1IbEozOw9b'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, local_filename, quiet=False)
    return joblib.load(local_filename)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 4. HEADER SECTION ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🌦️ WeatherAI: Australia Rain Predictor")
    st.markdown("""
    This app uses a **Random Forest Machine Learning model** to predict the likelihood of rain tomorrow 
    based on today's weather patterns across Australia.
    """)
with col2:
    st_lottie(lottie_sun, height=150, key="initial_sun")

st.markdown("---")

# --- 5. INPUT SECTION (The User Interface) ---
st.sidebar.header("⚙️ Weather Parameters")

with st.form("weather_form"):
    # Group inputs into Tabs for a cleaner look
    tab1, tab2, tab3 = st.tabs(["📍 Location & Date", "🌡️ Temperature & Sun", "💨 Wind & Humidity"])
    
    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            date_pick = st.date_input("Date")
            location = st.selectbox('Location', list(city_coords.keys()) + ['Albury', 'Cobar', 'Newcastle', 'Wollongong', 'AliceSprings']) # Add all your locations here
        with col_b:
            rain_today = st.selectbox('Did it rain today?', ['No', 'Yes'])
            
    with tab2:
        col_c, col_d = st.columns(2)
        with col_c:
            min_temp = st.slider('Min Temp (°C)', -10.0, 40.0, 15.0)
            max_temp = st.slider('Max Temp (°C)', -5.0, 50.0, 25.0)
            sunshine = st.slider('Sunshine (hours)', 0.0, 15.0, 8.0)
        with col_d:
            evaporation = st.number_input('Evaporation (mm)', 0.0, 100.0, 5.0)
            rainfall = st.number_input('Rainfall (mm)', 0.0, 400.0, 0.0)
            
    with tab3:
        col_e, col_f = st.columns(2)
        with col_e:
            wind_gust_speed = st.slider('Gust Speed (km/h)', 0.0, 135.0, 40.0)
            humidity_3pm = st.slider('Humidity 3pm (%)', 0, 100, 50)
            pressure_3pm = st.number_input('Pressure 3pm (hPa)', 900.0, 1100.0, 1012.0)
        with col_f:
            wind_dir = st.selectbox('Wind Gust Direction', ['N', 'S', 'E', 'W', 'NW', 'SE', 'SW', 'NE'])
            cloud_3pm = st.slider('Cloud 3pm (oktas)', 0, 9, 4)
            # Add hidden defaults for inputs we want to skip for the "Cleaner" UI
            # (In a real app, you'd add all sliders, but let's keep it simple for now)
    
    # Submit Button
    submitted = st.form_submit_button("🔮 Predict Weather")

# --- 6. PREDICTION LOGIC ---
if submitted:
    # 1. Date Engineering
    day, month, year = date_pick.day, date_pick.month, date_pick.year
    rain_today_enc = 1 if rain_today == 'Yes' else 0

    # 2. Construct Data Dictionary (Includes defaults for fields we didn't ask to keep UI clean)
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

    # 3. Handle Dummy Variables (Encoding)
    # Note: Ideally, you list ALL locations/directions. Here is a simplified handler.
    # It creates the dictionary keys for all 100+ columns
    
    # Load model columns (This trick ensures we have all 113 columns set to 0 initially)
    try:
        model_cols = model.feature_names_in_
    except:
        # Fallback list if feature_names_in_ isn't saved (Shouldn't happen with recent sklearn)
        st.error("Model features not found. Please re-save model with columns.")
        st.stop()
        
    # Create a DataFrame with all 0s
    df_predict = pd.DataFrame(0, index=[0], columns=model_cols)
    
    # Fill in the numerical values we have
    for col, val in input_data.items():
        if col in df_predict.columns:
            df_predict[col] = val
            
    # Handle Location & Wind One-Hot
    if f"Location_{location}" in df_predict.columns:
        df_predict[f"Location_{location}"] = 1
    if f"WindGustDir_{wind_dir}" in df_predict.columns:
        df_predict[f"WindGustDir_{wind_dir}"] = 1

    # 4. Prediction
    prediction = model.predict(df_predict)[0]
    probs = model.predict_proba(df_predict)[0]
    
    # --- 7. DISPLAY RESULTS ---
    st.markdown("### Analysis Result")
    
    result_col1, result_col2 = st.columns([1, 2])
    
    with result_col1:
        if prediction == 1:
            st.error("It will RAIN! ☔")
            st_lottie(lottie_rain, height=200, key="rain_anim")
        else:
            st.success("It will be SUNNY! ☀️")
            st_lottie(lottie_sun, height=200, key="sun_anim")
            
    with result_col2:
        # Gauge Chart using Plotly
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probs[1] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Rain Probability (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue" if prediction == 1 else "orange"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "lightcoral"}],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    # --- 8. MAP VISUALIZATION ---
    st.markdown(f"### 📍 Forecast Location: {location}")
    if location in city_coords:
        map_df = pd.DataFrame([city_coords[location]], columns=['lat', 'lon'])
        st.map(map_df)
    else:
        st.info("Map view available for major cities only in this demo.")
