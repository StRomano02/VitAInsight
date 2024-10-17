import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def load_model(filepath):
    """Load a model or scaler from a file."""
    try:
        return joblib.load(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")

def preprocess_input(data, scaler):
    """Preprocess input data using the given scaler."""
    return scaler.transform(data)

def load_data(file_path):
    """
    Load the dataset from the specified file path.
    """
    return pd.read_csv(file_path)


def display_image():
    st.image('images/health_dashboard_image.png', caption='Health Risk Prediction Dashboard', use_column_width=True)

def display_gauge_chart(risk_level, title="Risk Level"):
    # Definire i limiti del rischio
    gauge_value = 40 if risk_level == 0 else 90

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': gauge_value}}))

    st.plotly_chart(fig)
    


def display_gauge_chart_obesity(prediction_label):
    # Mappare il valore della predizione su un indicatore numerico (0-100)
    risk_mapping = {
        'Normal_Weight': 20,
        'Overweight_Level_I': 40,
        'Overweight_Level_II': 60,
        'Obesity_Type_I': 80,
        'Obesity_Type_II': 90,
        'Obesity_Type_III': 100
    }

    gauge_value = risk_mapping.get(prediction_label, 20)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value,
        title={'text': "Obesity Risk Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 20], 'color': "lightgreen"},
                {'range': [20, 40], 'color': "yellowgreen"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': gauge_value}}))

    st.plotly_chart(fig)

