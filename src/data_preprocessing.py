# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import load_data
import joblib
import os


def preprocess_data(df, target_column):
    """
    Preprocess the dataset by handling missing values and normalizing numerical features.
    Parameters:
        df (pd.DataFrame): The dataset to preprocess.
        target_column (str): The name of the target column in the dataset.
    Returns:
        X_train, X_test, y_train, y_test, scaler: Preprocessed training and test sets and the scaler used.
    """
    # Handle missing values or replace invalid values (e.g., 0 for glucose, BMI, etc.)
    for column in df.columns:
        if df[column].dtype in ['int64', 'float64'] and column != target_column:
            if df[column].min() == 0:  # Assuming 0 is an invalid value for these features
                df[column] = df[column].replace(0, df[column].mean())

    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def save_scaler(scaler, filepath):
    """
    Save the scaler to a specified file path.
    Parameters:
        scaler: The scaler to save.
        filepath (str): The file path where the scaler should be saved.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(scaler, filepath)
    print(f"Scaler saved to {filepath}")


if __name__ == "__main__":
    # Example usage for diabetes dataset
    df = load_data('data/raw/diabetes.csv')
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df, target_column='Outcome')

    # Save the scaler for future use
    save_scaler(scaler, 'data/preprocessed/diabetes_scaler.pkl')