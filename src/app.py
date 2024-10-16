# src/app.py

import streamlit as st
import pandas as pd
from data_preprocessing import preprocess_data, load_data
from model import train_model

# Carica i dati e addestra il modello
df = load_data('data/raw/diabetes.csv')  
X_train, X_test, y_train, y_test = preprocess_data(df)
model = train_model(X_train, y_train)

# Interfaccia utente
st.title("Diabete Risk Prediction")

pregnancies = st.number_input("Numero di gravidanze", min_value=0, max_value=20, value=0)
glucose = st.number_input("Livello di Glucosio", min_value=0.0, max_value=300.0, value=120.0)
blood_pressure = st.number_input("Pressione Sanguigna (mm Hg)", min_value=0.0, max_value=200.0, value=70.0)
skin_thickness = st.number_input("Spessore della pelle (mm)", min_value=0.0, max_value=100.0, value=20.0)
insulin = st.number_input("Livello di Insulina (µU/ml)", min_value=0.0, max_value=900.0, value=80.0)
bmi = st.number_input("Indice di Massa Corporea (BMI)", min_value=0.0, max_value=100.0, value=30.0)
diabetes_pedigree_function = st.number_input("Funzione Pedigree Diabetico", min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input("Età", min_value=0, max_value=120, value=25)

# Pulsante per fare la previsione
if st.button("Predict"):
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]], 
                              columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    prediction = model.predict(input_data)
    st.write(f"Risk: {prediction[0]}")
