import streamlit as st
import pandas as pd
from utils import load_model, preprocess_input

def diabetes_page():
    st.header("Diabetes Risk Prediction")

    # Inputs for the user
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, value=120.0)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)
    insulin = st.number_input("Insulin Level (µU/ml)", min_value=0.0, max_value=900.0, value=80.0)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=100.0, value=30.0)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=25)

    # Load model and scaler
    model = load_model('data/preprocessed/diabetes_model.pkl')
    scaler = load_model('data/preprocessed/diabetes_scaler.pkl')

    # Button to predict
    if st.button("Predict"):
        input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]],
                                  columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        # Preprocess input
        input_data_scaled = preprocess_input(input_data, scaler)

        # Make prediction
        prediction = model.predict(input_data_scaled)
        
        # Display result
        if prediction[0] == 1:
            st.markdown("<h2 style='color:red;'>High Risk of Diabetes</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:green;'>Low Risk of Diabetes</h2>", unsafe_allow_html=True)
