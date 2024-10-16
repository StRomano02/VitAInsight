import streamlit as st

def home_page():
    st.title("Health Risk Prediction Dashboard")
    st.write("Welcome to the health risk prediction platform.")
    
    # Inserisci un'immagine di benvenuto
    image_path = 'images/logo.png'  # Aggiorna con il percorso dell'immagine
    try:
        st.image(image_path, caption="Your health matters!", use_column_width=True)
    except FileNotFoundError:
        st.write("Image not found. Please check the path.")
    
    # Breve introduzione ai vari strumenti disponibili
    st.markdown("### Available Health Predictions")
    st.write("This platform offers predictive models for various health risks, including:")
    st.write("- **Diabetes Prediction**: Assess your risk of developing diabetes using the PIMA Indians Diabetes dataset.")
    st.write("- **Heart Disease Prediction**: Predict the likelihood of heart disease based on multiple factors.")
    st.write("- **Breast Cancer Prediction**: Predict whether a tumor is malignant or benign based on medical measurements.")
    st.write("- **Obesity Classification**: Classify your obesity level based on lifestyle and dietary habits.")
    
    # Istruzioni sull'uso della piattaforma
    st.markdown("### How to Use")
    st.write("Use the sidebar to navigate between different health prediction tools. Simply provide the required information and receive an analysis of your health risk.")

    # Link a pagine di analisi dei dataset
    st.markdown("### Dataset Analysis")
    st.write("If you are curious about the data behind these predictions, explore the **Dataset Analysis** section for each dataset.")