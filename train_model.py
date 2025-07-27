# train_model.py
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import numpy as np

def train_iris_model():
    """Train Iris classification model and save it."""
    # Load the iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    
    # Map target numbers to class names
    target_names = iris.target_names
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    
    # Save model
    joblib.dump(model, 'model.joblib')
    
    # Save metrics for CML reporting
    metrics = {
        'accuracy': float(accuracy),
        'test_samples': len(y_test),
        'train_samples': len(y_train),
        'n_features': X.shape[1]
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed report
    with open('classification_report.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Save confusion matrix data
    cm = confusion_matrix(y_test, y_pred)
    np.savetxt('confusion_matrix.csv', cm, delimiter=',', fmt='%d')
    
    print(f"Model trained successfully!")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model saved as model.joblib")
    
    return model, accuracy

if __name__ == "__main__":
    train_iris_model()
