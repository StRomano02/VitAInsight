# src/obesity_prediction.py

import streamlit as st
import pandas as pd
from utils import load_model, display_gauge_chart_obesity

def obesity_page():
    st.header("Obesity Risk Prediction")

    # Inputs for the user
    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=100, value=30)
    height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
    weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0)
    family_history = st.selectbox("Family History with Overweight", options=["yes", "no"])
    favc = st.selectbox("Frequent Consumption of High Caloric Food (FAVC)", options=["yes", "no"])
    fcvc = st.number_input("Frequency of Consumption of Vegetables (FCVC)", min_value=1, max_value=3, value=2)
    ncp = st.number_input("Number of Main Meals (NCP)", min_value=1, max_value=4, value=3)
    caec = st.selectbox("Consumption of Food between Meals (CAEC)", options=["no", "Sometimes", "Frequently", "Always"])
    smoke = st.selectbox("Do you Smoke?", options=["yes", "no"])
    ch2o = st.number_input("Daily Water Intake (liters)", min_value=1.0, max_value=4.0, value=2.0)
    scc = st.selectbox("Do you Monitor your Calorie Consumption? (SCC)", options=["yes", "no"])
    faf = st.number_input("Physical Activity Frequency (times per week)", min_value=0, max_value=7, value=0)
    tue = st.number_input("Time using technology devices (hours per day)", min_value=0, max_value=24, value=1)
    calc = st.selectbox("Consumption of Alcohol (CALC)", options=["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Transportation used", options=["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

    # Load model
    model = load_model('data/preprocessed/obesity_model.pkl')

    # Button to predict
    if st.button("Predict"):
        input_data = pd.DataFrame([[gender, age, height, weight, family_history, favc, fcvc, ncp, caec, smoke, ch2o, scc, faf, tue, calc, mtrans]],
                                  columns=['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS'])
        
        # Make prediction
        prediction = model.predict(input_data)[0]

        # Create a user-friendly message for each prediction
        prediction_mapping = {
            'Normal_Weight': "Your weight is in the normal range. Keep maintaining a healthy lifestyle!",
            'Overweight_Level_I': "You are at Overweight Level I. Consider making adjustments to your diet and increasing physical activity.",
            'Overweight_Level_II': "You are at Overweight Level II. It's advisable to consult a nutritionist or health professional.",
            'Obesity_Type_I': "You have Obesity Type I. A change in diet and increased exercise is highly recommended.",
            'Obesity_Type_II': "You have Obesity Type II. Please seek advice from a health professional to manage your weight effectively.",
            'Obesity_Type_III': "You have Obesity Type III. It is important to work closely with a healthcare provider to manage your condition."
        }

        # Utilizzo dopo la predizione per Obesity Prediction
        prediction_label = prediction[0]
        result_message = prediction_mapping.get(prediction, "Unknown prediction. Please consult a healthcare professional.")
        st.markdown(f"<h2 style='color:blue;'>Obesity Risk: {result_message}</h2>", unsafe_allow_html=True)
        display_gauge_chart_obesity(prediction_label)

    
