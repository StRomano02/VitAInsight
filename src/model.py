# src/model.py

from xgboost import XGBClassifier
from sklearn.metrics import classification_report

def train_model(X_train, y_train):
    """Allena il modello XGBoost."""
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Valuta il modello e stampa il report delle metriche."""
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    from data_preprocessing import load_data, preprocess_data

    df = load_data('data/raw/diabetes.csv') 
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
