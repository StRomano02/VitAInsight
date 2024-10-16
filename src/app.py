# src/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import preprocess_data, load_data
from model import train_model
import joblib
import os

# Load the dataset and train the model
df = load_data('data/raw/diabetes.csv')  
X_train, X_test, y_train, y_test = preprocess_data(df)
model = train_model(X_train, y_train)

# Load the scaler
scaler_path = 'data/preprocessed/scaler.pkl'
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
else:
    st.error("The scaler file was not found. Please run the preprocessing step first to generate 'scaler.pkl'.")

# Sidebar for page selection
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Personal Prediction", "Dataset Analysis"])

if page == "Personal Prediction":
    # Personal Prediction Section
    st.title("Diabetes Risk Prediction")

    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, value=120.0)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=70.0)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0)
    insulin = st.number_input("Insulin Level (µU/ml)", min_value=0.0, max_value=900.0, value=80.0)
    bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=100.0, value=30.0)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=25)

    # Button to predict
    if st.button("Predict"):
        input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]],
                                  columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        # Normalize the input data
        if 'scaler' in locals():
            input_data_scaled = scaler.transform(input_data)
        
            # Make prediction
            prediction = model.predict(input_data_scaled)
        
            # Display result
            if prediction[0] == 1:
                st.markdown("<h2 style='color:red;'>High Risk of Diabetes</h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='color:green;'>Low Risk of Diabetes</h2>", unsafe_allow_html=True)

            # Visualization of user's input compared to dataset
            st.subheader("Visualization of Your Data Compared to the Dataset")

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
                
        else:
            st.error("Scaler is not available. Please run the preprocessing step first.")

elif page == "Dataset Analysis":
    # Dataset Analysis Section
    st.title("Dataset Analysis")

    # Boxplot of Variables Compared to Outcome
    st.subheader("Boxplot of Variables Compared to Outcome")
    selected_variable = st.selectbox("Select a variable to view the boxplot", list(df.columns[:-1]))

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.boxplot(x='Outcome', y=selected_variable, data=df, ax=ax, palette='Set2', linewidth=2.5, width=0.5) 
    ax.set_title(f"Boxplot of {selected_variable} by Outcome")
    ax.set_xlabel("Outcome (0 = Low Risk, 1 = High Risk)")
    ax.set_ylabel(f"Value of {selected_variable}")
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap of Variables")
    st.pyplot(fig)
