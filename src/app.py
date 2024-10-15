# src/app.py

import streamlit as st
import pandas as pd
from data_preprocessing import preprocess_data, load_data
from model import train_model

# Carica i dati e addestra il modello
df = load_data('data/raw/diabetes.csv')  # Sostituisci con il tuo percorso di dataset
X_train, X_test, y_train, y_test = preprocess_data(df)
model = train_model(X_train, y_train)

# Interfaccia utente
st.title("Previsione del Rischio di Malattie Croniche")

age = st.number_input("Età", min_value=0, max_value=120)
bmi = st.number_input("Indice di Massa Corporea (BMI)", min_value=0.0, max_value=100.0)
# Aggiungi qui altri input necessari...

# Pulsante per fare la previsione
if st.button("Prevedi"):
    input_data = pd.DataFrame([[age, bmi]], columns=['age', 'bmi'])  # Aggiungi le altre colonne necessarie
    prediction = model.predict(input_data)
    st.write(f"Rischio: {prediction[0]}")
