# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
import sys
import os
import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Add parent directory to path to import the API
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a mock model for testing if the real one doesn't exist
def create_mock_model():
    """Create a simple mock model for testing."""
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(iris.data, iris.target)
    return model

# Ensure model exists BEFORE importing the app
if not os.path.exists('model.joblib'):
    mock_model = create_mock_model()
    joblib.dump(mock_model, 'model.joblib')

# ✅ Import after model is present so app loads model successfully
from iris_fastapi import app

# ✅ Test client fixture that handles FastAPI startup events
@pytest.fixture(scope="module")
def test_client():
    with TestClient(app) as client:
        yield client

class TestIrisAPI:
    """Test suite for Iris Classification API."""

    def test_root_endpoint(self, test_client):
        """Test the root endpoint returns welcome message."""
        response = test_client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert "Welcome" in response.json()["message"]

    def test_predict_endpoint_valid_input(self, test_client):
        """Test prediction endpoint with valid input."""
        test_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }

        response = test_client.post("/predict/", json=test_data)
        assert response.status_code == 200
        assert "predicted_class" in response.json()

        valid_classes = ["setosa", "versicolor", "virginica"]
        predicted_class = response.json()["predicted_class"]
        assert predicted_class in valid_classes

    def test_predict_endpoint_missing_field(self, test_client):
        """Test prediction endpoint with missing required field."""
        test_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4  # Missing petal_width
        }

        response = test_client.post("/predict/", json=test_data)
        assert response.status_code == 422

    def test_predict_endpoint_invalid_data_type(self, test_client):
        """Test prediction endpoint with invalid data types."""
        test_data = {
            "sepal_length": "invalid",
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }

        response = test_client.post("/predict/", json=test_data)
        assert response.status_code == 422

    def test_predict_endpoint_negative_values(self, test_client):
        """Test prediction endpoint with negative values."""
        test_data = {
            "sepal_length": -1.0,
            "sepal_width": -2.0,
            "petal_length": -3.0,
            "petal_width": -4.0
        }

        response = test_client.post("/predict/", json=test_data)
        assert response.status_code == 422

    def test_predict_endpoint_extreme_values(self, test_client):
        """Test prediction endpoint with extreme values."""
        test_data = {
            "sepal_length": 100.0,
            "sepal_width": 100.0,
            "petal_length": 100.0,
            "petal_width": 100.0
        }

        response = test_client.post("/predict/", json=test_data)
        assert response.status_code == 422

    def test_predict_multiple_samples(self, test_client):
        """Test prediction endpoint with multiple different samples."""
        test_samples = [
            {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            },
            {
                "sepal_length": 6.3,
                "sepal_width": 3.3,
                "petal_length": 4.7,
                "petal_width": 1.6
            },
            {
                "sepal_length": 7.3,
                "sepal_width": 2.9,
                "petal_length": 6.3,
                "petal_width": 1.8
            }
        ]

        predictions = []
        for sample in test_samples:
            response = test_client.post("/predict/", json=sample)
            assert response.status_code == 200
            predictions.append(response.json()["predicted_class"])

        valid_classes = ["setosa", "versicolor", "virginica"]
        for prediction in predictions:
            assert prediction in valid_classes

    def test_api_response_format(self, test_client):
        """Test that API response has correct format."""
        test_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }

        response = test_client.post("/predict/", json=test_data)
        assert response.status_code == 200

        json_response = response.json()
        assert isinstance(json_response, dict)
        assert "predicted_class" in json_response
        assert "confidence" in json_response
        assert "model_version" in json_response
        assert "timestamp" in json_response

# Integration test to verify model loading
def test_model_loading():
    """Test that the model can be loaded and makes reasonable predictions."""
    model = joblib.load('model.joblib')
    iris = load_iris()
    sample = iris.data[0:1]
    prediction = model.predict(sample)
    assert len(prediction) == 1
    assert prediction[0] in [0, 1, 2]

# Performance test
def test_api_response_time(test_client):
    """Test that API responds within reasonable time."""
    import time

    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    start_time = time.time()
    response = test_client.post("/predict/", json=test_data)
    response_time = time.time() - start_time

    assert response.status_code == 200
    assert response_time < 1.0  # Should respond within 1 second

