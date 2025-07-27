# generate_plots.py
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import json

def generate_confusion_matrix():
    """Generate confusion matrix plot."""
    # Load model and data
    model = joblib.load('model.joblib')
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    
    # Split data (same as training)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, 
                yticklabels=iris.target_names)
    plt.title('Confusion Matrix - Iris Classification')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Confusion matrix saved as confusion_matrix.png")

def generate_feature_importance():
    """Generate feature importance plot."""
    # Load model and data
    model = joblib.load('model.joblib')
    iris = load_iris()
    
    # Get feature importance
    importance = model.feature_importances_
    feature_names = iris.feature_names
    
    # Create DataFrame for plotting
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(importance_df['feature'], importance_df['importance'])
    
    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    plt.title('Feature Importance - Random Forest Classifier')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Feature importance plot saved as feature_importance.png")

def generate_data_distribution():
    """Generate data distribution plots."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    
    # Add target column
    df = X.copy()
    df['species'] = [iris.target_names[i] for i in y]
    
    # Create pairplot
    plt.figure(figsize=(12, 10))
    sns.pairplot(df, hue='species', diag_kind='hist')
    plt.suptitle('Iris Dataset - Feature Distributions', y=1.02)
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Data distribution plot saved as data_distribution.png")

def generate_model_performance_summary():
    """Generate a summary plot of model performance."""
    # Load validation results
    with open('validation_results.json', 'r') as f:
        results = json.load(f)
    
    # Create performance metrics bar chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [results['accuracy'], results['precision'], 
              results['recall'], results['f1_score']]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add threshold line for accuracy
    plt.axhline(y=results['min_accuracy_threshold'], color='red', 
                linestyle='--', alpha=0.7, label=f'Min Threshold ({results["min_accuracy_threshold"]})')
    
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Model performance summary saved as model_performance.png")

if __name__ == "__main__":
    generate_confusion_matrix()
    generate_feature_importance()
    generate_data_distribution()
    
    # Only generate performance summary if validation results exist
    try:
        generate_model_performance_summary()
    except FileNotFoundError:
        print("Validation results not found, skipping performance summary")
