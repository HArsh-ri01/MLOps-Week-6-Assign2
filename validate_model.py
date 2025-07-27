# validate_model.py
import joblib
import json
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def validate_model():
    """Validate the trained model and output metrics."""
    # Load the model
    model = joblib.load('model.joblib')
    
    # Load test data
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    
    # Use same split as training for consistency
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Define minimum acceptable performance
    min_accuracy = 0.85
    
    # Output metrics in a format suitable for CML
    print("=== MODEL VALIDATION RESULTS ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Test Samples: {len(y_test)}")
    
    # Model quality gate
    if accuracy >= min_accuracy:
        print(f"✅ Model PASSED validation (accuracy >= {min_accuracy})")
        validation_status = "PASSED"
    else:
        print(f"❌ Model FAILED validation (accuracy < {min_accuracy})")
        validation_status = "FAILED"
        exit(1)
    
    # Save validation results
    validation_results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'min_accuracy_threshold': min_accuracy,
        'validation_status': validation_status,
        'test_samples': len(y_test)
    }
    
    with open('validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"Validation results saved to validation_results.json")

if __name__ == "__main__":
    validate_model()
