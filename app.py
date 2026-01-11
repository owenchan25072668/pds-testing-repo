import streamlit as st
import pandas as pd
import joblib
import gdown
import os
import numpy as np

# --- 1. SETUP & STATE ---
st.set_page_config(page_title="Weather Predictor", page_icon="🌦️", layout="centered")

# Initialize Session State
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'inputs' not in st.session_state:
    st.session_state.inputs = {}

# --- 2. CONFIG DATA ---
LOCATIONS = [
    'Albury', 'BadgerysCreek', 'Cobar', 'CoffsHarbour', 'Moree',
    'Newcastle', 'NorahHead', 'NorfolkIsland', 'Penrith', 'Richmond',
    'Sydney', 'SydneyAirport', 'WaggaWagga', 'Williamtown', 'Wollongong',
    'Canberra', 'Tuggeranong', 'MountGinini', 'Ballarat', 'Bendigo',
    'Sale', 'MelbourneAirport', 'Melbourne', 'Mildura', 'Nhil',
    'Portland', 'Watsonia', 'Dartmoor', 'Brisbane', 'Cairns',
    'GoldCoast', 'Townsville', 'Adelaide', 'MountGambier', 'Nuriootpa',
    'Woomera', 'Albany', 'Witchcliffe', 'PearceRAAF', 'PerthAirport',
    'Perth', 'SalmonGums', 'Walpole', 'Hobart', 'Launceston',
    'AliceSprings', 'Darwin', 'Katherine', 'Uluru'
]

WIND_DIRECTIONS = [
    'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW', 'NA'
]

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

# --- 4. NAVIGATION HELPER ---
def go_to_step(new_step):
    st.session_state.step = new_step
    st.rerun()

# ==========================================
# STEP 1: LOCATION & DATE
# ==========================================
if st.session_state.step == 1:
    st.title("Step 1: Location & Date 📍")
    st.markdown("Start by selecting the location and date.")
    
    with st.form("step1_form"):
        col1, col2 = st.columns(2)
        with col1:
            date_pick = st.date_input("Date")
            location = st.selectbox('Location', LOCATIONS)
        with col2:
            rain_today = st.radio("Did it rain today?", ['No', 'Yes'], horizontal=True)
        
        if st.form_submit_button("Next Step ➡️", type="primary"):
            st.session_state.inputs['date'] = date_pick
            st.session_state.inputs['location'] = location
            st.session_state.inputs['rain_today'] = rain_today
            go_to_step(2)

# ==========================================
# STEP 2: ATMOSPHERE
# ==========================================
elif st.session_state.step == 2:
    st.title("Step 2: Atmosphere 🌡️")
    
    with st.form("step2_form"):
        # Row A: Temperature & Sunshine
        c1, c2, c3 = st.columns(3)
        with c1:
            min_temp = st.number_input('Min Temp (°C)', value=15.0)
            max_temp = st.number_input('Max Temp (°C)', value=25.0)
        with c2:
            rainfall = st.number_input('Rainfall (mm)', value=0.0)
            evaporation = st.number_input('Evaporation (mm)', value=5.0)
        with c3:
            sunshine = st.number_input('Sunshine (hours)', value=8.0)
        
        st.markdown("---")
        
        # Row B: Humidity, Pressure, Cloud (9am vs 3pm)
        c4, c5 = st.columns(2)
        with c4:
            st.markdown("**Morning (9am)**")
            hum_9 = st.slider('Humidity 9am (%)', 0, 100, 70)
            press_9 = st.number_input('Pressure 9am (hPa)', 900.0, 1100.0, 1015.0)
            cloud_9 = st.slider('Cloud 9am (0-9)', 0, 9, 4)
            temp_9 = st.number_input('Temp 9am (°C)', value=20.0)
            
        with c5:
            st.markdown("**Afternoon (3pm)**")
            hum_3 = st.slider('Humidity 3pm (%)', 0, 100, 50)
            press_3 = st.number_input('Pressure 3pm (hPa)', 900.0, 1100.0, 1012.0)
            cloud_3 = st.slider('Cloud 3pm (0-9)', 0, 9, 4)
            temp_3 = st.number_input('Temp 3pm (°C)', value=23.0)

        if st.form_submit_button("Next Step ➡️", type="primary"):
            st.session_state.inputs.update({
                'MinTemp': min_temp, 'MaxTemp': max_temp, 'Rainfall': rainfall,
                'Evaporation': evaporation, 'Sunshine': sunshine,
                'Humidity9am': hum_9, 'Pressure9am': press_9, 'Cloud9am': cloud_9, 'Temp9am': temp_9,
                'Humidity3pm': hum_3, 'Pressure3pm': press_3, 'Cloud3pm': cloud_3, 'Temp3pm': temp_3
            })
            go_to_step(3)

    if st.button("⬅️ Back"):
        go_to_step(1)

# ==========================================
# STEP 3: WIND DYNAMICS
# ==========================================
elif st.session_state.step == 3:
    st.title("Step 3: Wind Dynamics 💨")

    with st.form("step3_form"):
        st.markdown("### 🌪️ Wind Gust")
        wg_col1, wg_col2 = st.columns(2)
        with wg_col1:
            wg_dir = st.selectbox('Wind Gust Direction', WIND_DIRECTIONS)
        with wg_col2:
            wg_spd = st.slider('Wind Gust Speed (km/h)', 0.0, 150.0, 40.0)

        st.markdown("---")

        w_col1, w_col2 = st.columns(2)
        with w_col1:
            st.markdown("**Wind 9am**")
            w9_dir = st.selectbox('Wind Dir 9am', WIND_DIRECTIONS)
            w9_spd = st.slider('Wind Speed 9am (km/h)', 0.0, 100.0, 15.0)
        
        with w_col2:
            st.markdown("**Wind 3pm**")
            w3_dir = st.selectbox('Wind Dir 3pm', WIND_DIRECTIONS)
            w3_spd = st.slider('Wind Speed 3pm (km/h)', 0.0, 100.0, 20.0)

        submit_final = st.form_submit_button("🔮 Predict Result", type="primary")

    if st.button("⬅️ Back"):
        go_to_step(2)

    if submit_final:
        # Save inputs and GO TO STEP 4
        st.session_state.inputs.update({
            'WindGustDir': wg_dir, 'WindGustSpeed': wg_spd,
            'WindDir9am': w9_dir, 'WindSpeed9am': w9_spd,
            'WindDir3pm': w3_dir, 'WindSpeed3pm': w3_spd
        })
        go_to_step(4)

# ==========================================
# STEP 4: PREDICTION & RESULT
# ==========================================
elif st.session_state.step == 4:
    # 1. Gather all inputs
    inputs = st.session_state.inputs
    d = inputs['date']
    rain_today_enc = 1 if inputs['rain_today'] == 'Yes' else 0

    # 2. Prepare Data for Model (Numerical)
    final_input = {
        'Year': d.year, 'Month': d.month, 'Day': d.day,
        'RainToday': rain_today_enc,
        'MinTemp': inputs['MinTemp'], 'MaxTemp': inputs['MaxTemp'],
        'Rainfall': inputs['Rainfall'], 'Evaporation': inputs['Evaporation'],
        'Sunshine': inputs['Sunshine'],
        'Humidity9am': inputs['Humidity9am'], 'Humidity3pm': inputs['Humidity3pm'],
        'Pressure9am': inputs['Pressure9am'], 'Pressure3pm': inputs['Pressure3pm'],
        'Cloud9am': inputs['Cloud9am'], 'Cloud3pm': inputs['Cloud3pm'],
        'Temp9am': inputs['Temp9am'], 'Temp3pm': inputs['Temp3pm'],
        'WindGustSpeed': inputs['WindGustSpeed'], 
        'WindSpeed9am': inputs['WindSpeed9am'], 
        'WindSpeed3pm': inputs['WindSpeed3pm'],
        # These are used for logic below, not inserted directly into numerical cols
        'Location': inputs['location'],
        'WindGustDir': inputs['WindGustDir'],
        'WindDir9am': inputs['WindDir9am'],
        'WindDir3pm': inputs['WindDir3pm'],
    }

    # 3. Create DataFrame and Fill Cols
    try:
        model_cols = model.feature_names_in_
        df_predict = pd.DataFrame(0, index=[0], columns=model_cols)
        
        # Fill Numbers
        for col, val in final_input.items():
            if col in df_predict.columns and isinstance(val, (int, float)):
                df_predict[col] = val
        
        # Fill Categorical (One-Hot)
        loc = final_input['Location']
        if f"Location_{loc}" in df_predict.columns: df_predict[f"Location_{loc}"] = 1
        
        dirs = [('WindGustDir', final_input['WindGustDir']), 
                ('WindDir9am', final_input['WindDir9am']), 
                ('WindDir3pm', final_input['WindDir3pm'])]
        
        for prefix, direction in dirs:
            if direction != 'NA':
                col_name = f"{prefix}_{direction}"
                if col_name in df_predict.columns: df_predict[col_name] = 1

        # 4. Predict
        prediction = model.predict(df_predict)[0]
        probs = model.predict_proba(df_predict)[0]
        prob_rain = probs[1]

        # --- DISPLAY ---
        st.title("Prediction Report 📊")
        
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            if prediction == 1:
                st.error("☔ **RAIN EXPECTED**")
                st.metric("Confidence", f"{prob_rain*100:.1f}%")
            else:
                st.success("☀️ **NO RAIN**")
                st.metric("Confidence", f"{(1-prob_rain)*100:.1f}%")
            st.progress(prob_rain, text="Rain Probability")

        with col_res2:
            st.markdown("##### 📋 Input Summary")
            
            # Prepare Display Data (User Friendly)
            display_data = final_input.copy()
            display_data['RainToday'] = inputs['rain_today']
            display_data['Date'] = f"{d.day}/{d.month}/{d.year}"
            
            # Define Order
            desired_order = [
                'Date', 'Location', 'RainToday',
                'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                'Temp9am', 'Humidity9am', 'Cloud9am', 'WindDir9am', 'WindSpeed9am', 'Pressure9am',
                'Temp3pm', 'Humidity3pm', 'Cloud3pm', 'WindDir3pm', 'WindSpeed3pm', 'Pressure3pm',
                'WindGustDir', 'WindGustSpeed'
            ]
            
            table_rows = []
            for key in desired_order:
                if key in display_data:
                    table_rows.append({"Parameter": key, "Value": display_data[key]})
            
            st.dataframe(pd.DataFrame(table_rows), height=400, hide_index=True, width='stretch')

    except Exception as e:
        st.error(f"Error: {e}")

    # --- START OVER BUTTON ---
    st.markdown("---")
    if st.button("Start Over 🔄", type="primary"):
        st.session_state.step = 1
        st.session_state.inputs = {}
        st.rerun()
