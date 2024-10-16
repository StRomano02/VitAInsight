# src/app.py

import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
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
    
    # Fai la previsione
    prediction = model.predict(input_data)
    
    # Visualizza il risultato con uno stile più chiaro
    if prediction[0] == 1:
        st.markdown("<h2 style='color:red;'>Alto Rischio di Diabete</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color:green;'>Basso Rischio di Diabete</h2>", unsafe_allow_html=True)

     # Creazione di grafici per ogni variabile
    st.subheader("Visual¡zation of your data compared to the dataset")

    variables = {
        "Pregnancies": pregnancies, 
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi
    }
    
    
    col1, col2 = st.columns(2) 

    for idx, (var_name, user_value) in enumerate(variables.items()):
        fig, ax = plt.subplots(figsize=(4, 2), dpi=50)  
        ax.hist(df[var_name], bins=30, alpha=0.7, label=var_name)
        ax.axvline(user_value, color='red', linestyle='--', label=f'Your {var_name}: {user_value}')
        ax.set_title(f"Distribution of levels of {var_name}")
        ax.set_xlabel(f"Levels of {var_name}")
        ax.set_ylabel("Frquence")
        ax.legend()
        if idx % 2 == 0:
            col1.pyplot(fig)
        else:
            col2.pyplot(fig)

