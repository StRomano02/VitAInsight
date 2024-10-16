import streamlit as st
import pandas as pd
from utils import load_model

def breast_cancer_page():
    st.header("Breast Cancer Risk Prediction")

    # Inputs for the user (31 features)
    radius_mean = st.number_input("Radius Mean", min_value=0.0, max_value=50.0, value=14.0)
    texture_mean = st.number_input("Texture Mean", min_value=0.0, max_value=50.0, value=20.0)
    perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, max_value=200.0, value=80.0)
    area_mean = st.number_input("Area Mean", min_value=0.0, max_value=2500.0, value=500.0)
    smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, max_value=1.0, value=0.1)
    compactness_mean = st.number_input("Compactness Mean", min_value=0.0, max_value=1.0, value=0.2)
    concavity_mean = st.number_input("Concavity Mean", min_value=0.0, max_value=1.0, value=0.3)
    concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, max_value=1.0, value=0.1)
    symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, max_value=1.0, value=0.2)
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0, max_value=1.0, value=0.05)
    radius_se = st.number_input("Radius SE", min_value=0.0, max_value=10.0, value=1.1)
    texture_se = st.number_input("Texture SE", min_value=0.0, max_value=10.0, value=1.0)
    perimeter_se = st.number_input("Perimeter SE", min_value=0.0, max_value=100.0, value=8.0)
    area_se = st.number_input("Area SE", min_value=0.0, max_value=500.0, value=40.0)
    smoothness_se = st.number_input("Smoothness SE", min_value=0.0, max_value=1.0, value=0.005)
    compactness_se = st.number_input("Compactness SE", min_value=0.0, max_value=1.0, value=0.03)
    concavity_se = st.number_input("Concavity SE", min_value=0.0, max_value=1.0, value=0.04)
    concave_points_se = st.number_input("Concave Points SE", min_value=0.0, max_value=1.0, value=0.02)
    symmetry_se = st.number_input("Symmetry SE", min_value=0.0, max_value=1.0, value=0.02)
    fractal_dimension_se = st.number_input("Fractal Dimension SE", min_value=0.0, max_value=1.0, value=0.003)
    radius_worst = st.number_input("Radius Worst", min_value=0.0, max_value=50.0, value=25.0)
    texture_worst = st.number_input("Texture Worst", min_value=0.0, max_value=50.0, value=17.0)
    perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0, max_value=200.0, value=184.0)
    area_worst = st.number_input("Area Worst", min_value=0.0, max_value=2500.0, value=2019.0)
    smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0, max_value=1.0, value=0.16)
    compactness_worst = st.number_input("Compactness Worst", min_value=0.0, max_value=1.0, value=0.66)
    concavity_worst = st.number_input("Concavity Worst", min_value=0.0, max_value=1.0, value=0.71)
    concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0, max_value=1.0, value=0.27)
    symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0, max_value=1.0, value=0.46)
    fractal_dimension_worst = st.number_input("Fractal Dimension Worst", min_value=0.0, max_value=1.0, value=0.12)

    # Load model
    model = load_model('data/preprocessed/breast_cancer_model.pkl')

    # Button to predict
    if st.button("Predict"):
        input_data = pd.DataFrame([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                                    compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
                                    radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se,
                                    concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
                                    perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst,
                                    concave_points_worst, symmetry_worst, fractal_dimension_worst]],
                                  columns=['Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Smoothness Mean',
                                           'Compactness Mean', 'Concavity Mean', 'Concave Points Mean', 'Symmetry Mean', 'Fractal Dimension Mean',
                                           'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE', 'Concavity SE',
                                           'Concave Points SE', 'Symmetry SE', 'Fractal Dimension SE', 'Radius Worst', 'Texture Worst',
                                           'Perimeter Worst', 'Area Worst', 'Smoothness Worst', 'Compactness Worst', 'Concavity Worst',
                                           'Concave Points Worst', 'Symmetry Worst', 'Fractal Dimension Worst'])

        # Make prediction
        prediction = model.predict(input_data)

        # Display result
        if prediction[0] == 1:
            st.markdown("<h2 style='color:red;'>High Risk of Breast Cancer</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:green;'>Low Risk of Breast Cancer</h2>", unsafe_allow_html=True)
