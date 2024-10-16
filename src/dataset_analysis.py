import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_model, load_data
import streamlit as st
from diabetes_prediction import diabetes_page
from heart_disease_prediction import heart_disease_page
from breast_cancer_prediction import breast_cancer_page
from obesity_prediction import obesity_page
from home_page import home_page


def dataset_analysis_page():
    st.title("Dataset Analysis")
    st.write("Explore and compare the dataset features for each health risk.")

    # Seleziona il dataset da visualizzare
    dataset_choice = st.selectbox("Choose a dataset to analyze", ["Diabetes", "Heart Disease", "Breast Cancer", "Obesity"])

    # Carica il dataset corrispondente
    if dataset_choice == "Diabetes":
        df = load_data('data/raw/diabetes.csv')
        target_column = 'Outcome'
        description = "PIMA Indians Diabetes Database. The outcome variable indicates whether a person has diabetes (1) or not (0)."
    elif dataset_choice == "Heart Disease":
        df = load_data('data/raw/heart_disease.csv')
        target_column = 'HeartDisease'
        description = "Heart Disease Dataset. The target variable 'HeartDisease' indicates the presence (1) or absence (0) of heart disease."
    elif dataset_choice == "Breast Cancer":
        df = load_data('data/raw/breast_cancer.csv')
        target_column = 'diagnosis'
        description = "Breast Cancer Wisconsin Dataset. The target variable 'diagnosis' indicates whether a tumor is malignant (M = 1) or benign (B = 0)."
        df[target_column] = df[target_column].map({'M': 1, 'B': 0})
        if 'id' in df.columns:
            df = df.drop('id', axis=1)
        if 'Unnamed: 32' in df.columns:
            df = df.drop('Unnamed: 32', axis=1)
    elif dataset_choice == "Obesity":
        df = load_data('data/raw/obesity.csv')
        target_column = 'NObeyesdad'
        description = "Obesity Dataset. The target variable 'NObeyesdad' indicates the obesity classification."
    
    st.write(description)

    # Boxplot di variabili rispetto all'outcome
    st.subheader(f"Boxplot of Variables Compared to {target_column}")
    selected_variable = st.selectbox("Select a variable to view the boxplot", list(df.columns.difference([target_column])))

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x=target_column, y=selected_variable, data=df, ax=ax, palette='Set2', linewidth=2.5, width=0.5)
    ax.set_title(f"Boxplot of {selected_variable} by {target_column}")
    ax.set_xlabel(f"{target_column} (0 = Low Risk, 1 = High Risk)")
    ax.set_ylabel(f"Value of {selected_variable}")
    st.pyplot(fig)

    # Heatmap di correlazione
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    ax.set_title(f"Correlation Heatmap of Variables in {dataset_choice} Dataset")
    st.pyplot(fig)

    # Distribuzione delle variabili e confronto con l'input dell'utente (se disponibile)
    st.subheader("Visualization of Dataset Distribution")
    col1, col2 = st.columns(2)

    for idx, column in enumerate(df.columns.difference([target_column])):
        fig, ax = plt.subplots(figsize=(4, 2), dpi=50)
        ax.hist(df[column], bins=30, alpha=0.7, label=column)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(f"{column}")
        ax.set_ylabel("Frequency")
        ax.legend()
        if idx % 2 == 0:
            col1.pyplot(fig)
        else:
            col2.pyplot(fig)

# Aggiornare l'app per includere la nuova pagina
page = st.sidebar.radio("Choose an option:", ["Home", "Diabetes Prediction", "Heart Disease Prediction", "Breast Cancer Prediction", "Obesity Prediction", "Dataset Analysis"])

if page == "Home":
    home_page()
elif page == "Diabetes Prediction":
    diabetes_page()
elif page == "Heart Disease Prediction":
    heart_disease_page()
elif page == "Breast Cancer Prediction":
    breast_cancer_page()
elif page == "Obesity Prediction":
    obesity_page()
elif page == "Dataset Analysis":
    dataset_analysis_page()
