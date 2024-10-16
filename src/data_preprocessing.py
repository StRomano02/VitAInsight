# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(file_path):
    """Load the dataset."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data by handling missing values and normalizing numerical features."""
    
    df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())
    df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())
    df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].mean())
    df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].mean())
    df['BMI'] = df['BMI'].replace(0, df['BMI'].mean())

    # Separate features and target
    X = df.drop('Outcome', axis=1) 
    y = df['Outcome']

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Ensure the directory exists
    os.makedirs('data/preprocessed', exist_ok=True)

    # Save the scaler for future use
    joblib.dump(scaler, 'data/preprocessed/scaler.pkl')
    print("Scaler saved correctly as 'data/preprocessed/scaler.pkl'")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data('data/raw/diabetes.csv') 
    X_train, X_test, y_train, y_test = preprocess_data(df)
