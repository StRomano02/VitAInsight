# src/heart_disease_prediction.py

import streamlit as st
import pandas as pd
from utils import load_model

def heart_disease_page():
    st.header("Heart Disease Risk Prediction")

    # Inputs for the user
    age = st.number_input("Age", min_value=0, max_value=120, value=40)
    sex = st.selectbox("Sex", options=["M", "F"])
    chest_pain_type = st.selectbox("Chest Pain Type", options=["TA", "ATA", "NAP", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=250, value=140)
    cholesterol = st.number_input("Cholesterol Level (mg/dl)", min_value=0, max_value=600, value=289)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["0", "1"])
    resting_ecg = st.selectbox("Resting ECG Results", options=["Normal", "ST", "LVH"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, value=172)
    exercise_angina = st.selectbox("Exercise Induced Angina", options=["Y", "N"])
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=0.0)
    st_slope = st.selectbox("ST Slope", options=["Up", "Flat", "Down"])

    # Load model
    model = load_model('data/preprocessed/heart_disease_model.pkl')

    # Button to predict
    if st.button("Predict"):
        input_data = pd.DataFrame([[age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]],
                                  columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope'])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Display result
        if prediction[0] == 1:
            st.markdown("<h2 style='color:red;'>High Risk of Heart Disease</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:green;'>Low Risk of Heart Disease</h2>", unsafe_allow_html=True)

