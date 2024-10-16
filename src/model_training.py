import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_diabetes_model():
    # Load and preprocess the dataset
    df = pd.read_csv('data/raw/diabetes.csv')
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train_scaled, y_train)

    # Save model and scaler
    joblib.dump(model, 'data/preprocessed/diabetes_model.pkl')
    joblib.dump(scaler, 'data/preprocessed/diabetes_scaler.pkl')


def train_heart_disease_model():
    # Load the dataset
    df = pd.read_csv('data/raw/heart_disease.csv')
    
    # Print column names to verify the target column name
    print("Column names in heart disease dataset:", df.columns)
    
    # Set the target column
    target_column = 'HeartDisease'

    # Convert categorical features to numerical using one-hot encoding
    categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']

    # Column transformer to handle categorical and numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # Create a pipeline with preprocessor and model
    from sklearn.ensemble import RandomForestClassifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])

    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'data/preprocessed/heart_disease_model.pkl')
    print("Heart disease model saved successfully.")


def train_breast_cancer_model():
    # Load the dataset
    df = pd.read_csv('data/raw/breast_cancer.csv')
    
    # Print column names to verify the target column name
    print("Column names in breast cancer dataset:", df.columns)
    
    # Set the target column
    target_column = 'diagnosis'
    
    # Convert the target column from categorical to numerical ('M' -> 1, 'B' -> 0)
    df[target_column] = df[target_column].map({'M': 1, 'B': 0})
        
    # Drop the 'id' column as it is not useful for prediction
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
        
    # Drop 'Unnamed: 32' if it exists
    if 'Unnamed: 32' in df.columns:
        df = df.drop('Unnamed: 32', axis=1)
        
    # Print the columns to verify
    print("Columns after dropping 'id' and converting target:", df.columns)

    # Verify if target_column is correct
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not found in dataset. Available columns: {df.columns}")
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    model.fit(X_train_scaled, y_train)

    # Save model and scaler
    joblib.dump(model, 'data/preprocessed/breast_cancer_model.pkl')
    joblib.dump(scaler, 'data/preprocessed/breast_cancer_scaler.pkl')
    print("Breast cancer model and scaler saved successfully.")


def train_obesity_model():
    # Load the dataset
    df = pd.read_csv('data/raw/obesity.csv')
    
    # Print column names to verify the target column name
    print("Column names in obesity dataset:", df.columns)
    
    # Set the target column
    target_column = 'NObeyesdad'

    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # One-hot encoding for categorical features
    categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    numeric_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    
    # Column transformer to handle categorical and numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    
    # Create a pipeline with preprocessor and model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'data/preprocessed/obesity_model.pkl')
    print("Obesity model saved successfully.")


def main():
    train_diabetes_model()
    train_heart_disease_model()
    train_breast_cancer_model()
    train_obesity_model()
    print("All models trained and saved successfully.")

if __name__ == "__main__":
    main()
