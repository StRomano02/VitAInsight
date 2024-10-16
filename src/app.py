import streamlit as st

# Importa le funzioni per le varie pagine
from diabetes_prediction import diabetes_page
from heart_disease_prediction import heart_disease_page
from breast_cancer_prediction import breast_cancer_page
from obesity_prediction import obesity_page
from home_page import home_page
from dataset_analysis import dataset_analysis_page

# Sidebar for navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Choose an option:", [
    "Home", 
    "Diabetes Prediction", 
    "Heart Disease Prediction", 
    "Breast Cancer Prediction", 
    "Obesity Prediction", 
    "Data Analysis"
])

if selection == "Home":
    home_page()  # Calls the home page function

elif selection == "Diabetes Prediction":
    diabetes_page()  # Calls the diabetes prediction page function

elif selection == "Heart Disease Prediction":
    heart_disease_page()  # Calls the heart disease prediction page function

elif selection == "Breast Cancer Prediction":
    breast_cancer_page()  # Calls the breast cancer prediction page function

elif selection == "Obesity Prediction":
    obesity_page()  # Calls the obesity prediction page function

elif selection == "Data Analysis":
    dataset_analysis_page()  # Calls the dataset analysis page function
