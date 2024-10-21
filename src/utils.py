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
    


