import streamlit as st
from PIL import Image
from diabetes_prediction import diabetes_page
from heart_disease_prediction import heart_disease_page
from breast_cancer_prediction import breast_cancer_page
from obesity_assessment import obesity_page
from dataset_analysis import dataset_analysis_page

# Load images for the homepage
diabetes_image = Image.open('images/diabetes.png')
heart_image = Image.open('images/heart_disease.png')
cancer_image = Image.open('images/breast_cancer.png')
obesity_image = Image.open('images/obesity.png')
dataset_analysis_image = Image.open('images/dataset_analysis.png')

# Sidebar per la navigazione
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Choose an option:", [
    "Home",
    "Diabetes Prediction",
    "Heart Disease Prediction",
    "Breast Cancer Prediction",
    "Obesity Assessment",
    "Data Analysis"
])

# Home Page
def home_page():
    st.title("VitAInsight: Health Risk Prediction Dashboard")
    st.write("Welcome to the health risk prediction platform. Choose an option below to begin:")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Diabetes Prediction"):
            st.experimental_set_query_params(page="Diabetes Prediction")
        st.image(diabetes_image, use_column_width=True, caption="Diabetes Prediction")

    with col2:
        if st.button("Heart Disease Prediction"):
            st.experimental_set_query_params(page="Heart Disease Prediction")
        st.image(heart_image, use_column_width=True, caption="Heart Disease Prediction")

    col3, col4 = st.columns(2)

    with col3:
        if st.button("Breast Cancer Prediction"):
            st.experimental_set_query_params(page="Breast Cancer Prediction")
        st.image(cancer_image, use_column_width=True, caption="Breast Cancer Prediction")

    with col4:
        if st.button("Obesity Assessment"):
            st.experimental_set_query_params(page="Obesity Assessment")
        st.image(obesity_image, use_column_width=True, caption="Obesity Assessment")

    col5, col6 = st.columns(2)

    with col5:
        if st.button("Dataset Analysis"):
            st.experimental_set_query_params(page="Data Analysis")
        st.image(dataset_analysis_image, use_column_width=True, caption="Dataset Analysis")

# Chiamata della pagina selezionata
if selection == "Home":
    home_page()
elif selection == "Diabetes Prediction":
    diabetes_page()
elif selection == "Heart Disease Prediction":
    heart_disease_page()
elif selection == "Breast Cancer Prediction":
    breast_cancer_page()
elif selection == "Obesity Assessment":
    obesity_page()
elif selection == "Data Analysis":
    dataset_analysis_page()
