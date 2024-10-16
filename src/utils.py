import joblib
import pandas as pd

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