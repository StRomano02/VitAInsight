# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Carica il dataset."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocessa i dati, gestendo valori mancanti e normalizzando le variabili numeriche."""
    
    # df.fillna(df.mean(), inplace=True) 
    # non utilizziamo questa funzione perchè per le pregnancies non è 
    # corretto utilizzare come valore NaN lo 0 e sostituirlo con la media
    
    df['Glucose'].replace(0, df['Glucose'].mean(), inplace=True)
    df['BloodPressure'].replace(0, df['BloodPressure'].mean(), inplace=True)
    df['SkinThickness'].replace(0, df['SkinThickness'].mean(), inplace=True)
    df['Insulin'].replace(0, df['Insulin'].mean(), inplace=True)
    df['BMI'].replace(0, df['BMI'].mean(), inplace=True)
    
    # Codifica delle variabili categoriche (one-hot encoding)
    # anche se credo siano tutti valori numerici
    df = pd.get_dummies(df)

    # Separazione delle feature e del target
    X = df.drop('Outcome', axis=1) 
    y = df['Outcome']

    # Divisione in set di addestramento e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizzazione delle features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Esempio di utilizzo
    df = load_data('data/raw/diabetes.csv') 
    X_train, X_test, y_train, y_test = preprocess_data(df)
