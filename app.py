import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np

# --- 1. SETUP & STATE ---
st.set_page_config(page_title="Weather Predictor", page_icon="🌦️", layout="centered")

# Initialize the "Backpack" (Session State)
# We use this to track which 'step' the user is on
if 'step' not in st.session_state:
    st.session_state.step = 1

# We also need a place to store the inputs as we move between pages
if 'inputs' not in st.session_state:
    st.session_state.inputs = {}

# --- 2. MODEL LOADING ---
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

# --- 3. HELPER FUNCTIONS FOR NAVIGATION ---
def go_to_step(new_step):
    st.session_state.step = new_step
    # Rerun acts like a "Refresh" to show the new page immediately
    st.rerun()

# --- 4. STEP 1: LOCATION & DATE ---
if st.session_state.step == 1:
    st.title("Step 1: Location & Time 📍")
    st.markdown("Let's start with where and when.")
    
    with st.form("step1_form"):
        col1, col2 = st.columns(2)
        with col1:
            date_pick = st.date_input("Select Date")
            location = st.selectbox('City / Location', 
                                  ['Sydney', 'Melbourne', 'Brisbane', 'Perth', 'Adelaide', 'Canberra', 'Hobart', 'Darwin', 'AliceSprings', 'Cobar', 'NorfolkIsland'])
        with col2:
            rain_today = st.radio("Did it rain today?", ['No', 'Yes'], horizontal=True)
        
        # When clicking Next, we save data and move to step 2
        if st.form_submit_button("Next Step ➡️", type="primary"):
            # Save inputs to our 'backpack'
            st.session_state.inputs['date'] = date_pick
            st.session_state.inputs['location'] = location
            st.session_state.inputs['rain_today'] = rain_today
            
            # Move to next page
            go_to_step(2)

# --- 5. STEP 2: TEMPERATURE & CONDITIONS ---
elif st.session_state.step == 2:
    st.title("Step 2: Temperature Conditions 🌡️")
    st.markdown("What are the thermal conditions?")
    
    with st.form("step2_form"):
        col1, col2 = st.columns(2)
        with col1:
            min_temp = st.slider('Min Temp (°C)', -10.0, 40.0, 15.0)
            max_temp = st.slider('Max Temp (°C)', -5.0, 50.0, 25.0)
            sunshine = st.slider('Sunshine (hours)', 0.0, 15.0, 8.0)
        with col2:
            evaporation = st.number_input('Evaporation (mm)', 0.0, 100.0, 5.0)
            rainfall = st.number_input('Rainfall (mm)', 0.0, 400.0, 0.0)

        # Navigation Buttons
        c1, c2 = st.columns([1, 1])
        with c1:
            # We use a button (not form_submit) for Back to avoid validation issues
            # But inside a form, everything must be a submit button or outside.
            # Trick: Use form_submit for both, handle logic below.
            submitted = st.form_submit_button("Next Step ➡️", type="primary")
        
    if submitted:
        # Save inputs
        st.session_state.inputs['min_temp'] = min_temp
        st.session_state.inputs['max_temp'] = max_temp
        st.session_state.inputs['sunshine'] = sunshine
        st.session_state.inputs['evaporation'] = evaporation
        st.session_state.inputs['rainfall'] = rainfall
        go_to_step(3)

    # Back button outside the form to prevent auto-submission confusion
    if st.button("⬅️ Back"):
        go_to_step(1)

# --- 6. STEP 3: WIND & FINAL PREDICTION ---
elif st.session_state.step == 3:
    st.title("Step 3: Wind & Pressure 💨")
    st.markdown("Final details before prediction.")

    with st.form("step3_form"):
        col1, col2 = st.columns(2)
        with col1:
            wind_gust_speed = st.slider('Wind Gust Speed (km/h)', 0.0, 135.0, 40.0)
            humidity_3pm = st.slider('Humidity 3pm (%)', 0, 100, 50)
            pressure_3pm = st.number_input('Pressure 3pm (hPa)', 900.0, 1100.0, 1012.0)
        with col2:
            wind_dir = st.selectbox('Wind Direction', ['N', 'S', 'E', 'W', 'NW', 'SE', 'SW', 'NE'])
            cloud_3pm = st.slider('Cloud Cover 3pm (0-9)', 0, 9, 4)

        submit_final = st.form_submit_button("🔮 Predict Result", type="primary")

    if st.button("⬅️ Back"):
        go_to_step(2)

    # --- PREDICTION LOGIC ---
    if submit_final:
        # Retrieve all data from the 'backpack'
        inputs = st.session_state.inputs
        
        # Date Engineering
        d = inputs['date']
        day, month, year = d.day, d.month, d.year
        rain_today_enc = 1 if inputs['rain_today'] == 'Yes' else 0

        # Construct Final Input Dictionary
        # Merging Session State data (Step 1 & 2) with Current Form data (Step 3)
        final_data = {
            'MinTemp': inputs['min_temp'], 'MaxTemp': inputs['max_temp'], 
            'Rainfall': inputs['rainfall'], 'Evaporation': inputs['evaporation'], 
            'Sunshine': inputs['sunshine'],
            'WindGustSpeed': wind_gust_speed, 'WindSpeed9am': 15.0, 'WindSpeed3pm': 20.0,
            'Humidity9am': 70, 'Humidity3pm': humidity_3pm,
            'Pressure9am': 1015.0, 'Pressure3pm': pressure_3pm,
            'Cloud9am': 4, 'Cloud3pm': cloud_3pm,
            'Temp9am': 20.0, 'Temp3pm': 23.0,
            'RainToday_Yes': rain_today_enc, 'Year': year, 'Month': month, 'Day': day
        }

        # Align with Model Columns
        try:
            model_cols = model.feature_names_in_
            df_predict = pd.DataFrame(0, index=[0], columns=model_cols)
        except:
            st.error("Model error. Please check .joblib file.")
            st.stop()

        # Fill Numerical
        for col, val in final_data.items():
            if col in df_predict.columns:
                df_predict[col] = val

        # Fill One-Hot
        loc = inputs['location']
        if f"Location_{loc}" in df_predict.columns:
            df_predict[f"Location_{loc}"] = 1
        if f"WindGustDir_{wind_dir}" in df_predict.columns:
            df_predict[f"WindGustDir_{wind_dir}"] = 1

        # Prediction
        prediction = model.predict(df_predict)[0]
        probs = model.predict_proba(df_predict)[0]
        prob_rain = probs[1]

        # --- DISPLAY RESULT ---
        st.markdown("---")
        st.subheader("Prediction Report")
        
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            if prediction == 1:
                st.error("☔ **RAIN EXPECTED**")
            else:
                st.success("☀️ **NO RAIN**")
        
        with res_col2:
            st.progress(prob_rain, text=f"Probability of Rain: {prob_rain*100:.1f}%")
        
        # Option to Start Over
        if st.button("Start Over 🔄"):
            st.session_state.step = 1
            st.session_state.inputs = {}
            st.rerun()
