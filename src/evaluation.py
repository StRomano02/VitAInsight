# src/evaluation.py

from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred):
    """Visualizza la matrice di confusione."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def calculate_accuracy(y_true, y_pred):
    """Calcola e restituisce l'accuratezza."""
    return accuracy_score(y_true, y_pred)

if __name__ == "__main__":
    from model import train_model, evaluate_model
    from data_preprocessing import load_data, preprocess_data

    df = load_data('data/raw/diabetes.csv')  # Sostituisci con il tuo percorso di dataset
    X_train, X_test, y_train, y_test = preprocess_data(df)

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = calculate_accuracy(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    plot_confusion_matrix(y_test, y_pred)
