# iris_fastapi.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app with metadata
app = FastAPI(
    title="Iris Classifier API 🌸",
    description="A machine learning API for classifying Iris flower species using Random Forest",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global variables
model = None
model_metadata = {}

# Input schema with validation
class IrisInput(BaseModel):
    sepal_length: float = Field(..., ge=0, le=20, description="Sepal length in cm")
    sepal_width: float = Field(..., ge=0, le=20, description="Sepal width in cm") 
    petal_length: float = Field(..., ge=0, le=20, description="Petal length in cm")
    petal_width: float = Field(..., ge=0, le=20, description="Petal width in cm")
    
    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

# Response schema
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(..., description="Prediction confidence score")
    model_version: str = Field(..., description="Model version used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool
    timestamp: str
    version: str

# Load model on startup
@app.on_event("startup")
async def load_model():
    """Load the trained model on application startup."""
    global model, model_metadata
    
    try:
        model_path = "model.joblib"
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model = joblib.load(model_path)
        model_metadata = {
            "version": "1.0.0",
            "algorithm": "RandomForestClassifier",
            "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
            "classes": ["setosa", "versicolor", "virginica"],
            "loaded_at": datetime.now().isoformat()
        }
        
        logger.info(f"Model loaded successfully: {model_metadata}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e

# Health check endpoint
@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API is running and model is loaded.
    """
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        message="Welcome to the Iris Classifier API! 🌸" if model is not None else "Model not loaded",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

# Prediction endpoint
@app.post("/predict/", response_model=PredictionResponse, tags=["Prediction"])
async def predict_species(data: IrisInput):
    """
    Predict the Iris species based on flower measurements.
    
    - **sepal_length**: Length of the sepal in centimeters
    - **sepal_width**: Width of the sepal in centimeters  
    - **petal_length**: Length of the petal in centimeters
    - **petal_width**: Width of the petal in centimeters
    
    Returns the predicted species and confidence score.
    """
    if model is None:
        logger.error("Model not loaded")
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame (required for the model)
        input_dict = {
            "sepal length (cm)": data.sepal_length,
            "sepal width (cm)": data.sepal_width,
            "petal length (cm)": data.petal_length,
            "petal width (cm)": data.petal_width,
        }
        input_data = pd.DataFrame([input_dict])

        
        # Log the input for monitoring
        logger.info(f"Prediction request: {input_dict}")
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Get prediction probabilities for confidence
        prediction_proba = model.predict_proba(input_data)[0]
        confidence = float(max(prediction_proba))
        
        # Map numeric prediction to class name
        class_names = model_metadata["classes"]
        predicted_class = class_names[prediction]
        
        response = PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            model_version=model_metadata["version"],
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction response: {response.dict()}")
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Model information endpoint
@app.get("/model/info", response_model=Dict[str, Any], tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return model_metadata

# Batch prediction endpoint
@app.post("/predict/batch/", tags=["Prediction"])
async def predict_batch(data: list[IrisInput]):
    """
    Predict the Iris species for multiple samples at once.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if len(data) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100 samples")
    
    try:
        # Convert all inputs to DataFrame
        input_df = pd.DataFrame([sample.dict() for sample in data])
        
        # Make predictions
        predictions = model.predict(input_df)
        prediction_probas = model.predict_proba(input_df)
        
        # Prepare response
        results = []
        class_names = model_metadata["classes"]
        timestamp = datetime.now().isoformat()
        
        for i, (pred, proba) in enumerate(zip(predictions, prediction_probas)):
            results.append({
                "sample_id": i,
                "predicted_class": class_names[pred],
                "confidence": float(max(proba)),
                "model_version": model_metadata["version"],
                "timestamp": timestamp
            })
        
        logger.info(f"Batch prediction completed for {len(data)} samples")
        return {"predictions": results, "total_samples": len(data)}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Model metrics endpoint (if available)
@app.get("/model/metrics", tags=["Model"])
async def get_model_metrics():
    """
    Get model performance metrics if available.
    """
    try:
        # Try to load metrics from validation
        metrics_file = "validation_results.json"
        if os.path.exists(metrics_file):
            import json
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            return metrics
        else:
            return {"message": "Model metrics not available"}
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}")
        return {"error": "Could not load model metrics"}

# Custom exception handler
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    General exception handler for unhandled errors.
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    return HTTPException(status_code=500, detail="Internal server error")

# Add middleware for request logging
@app.middleware("http")
async def log_requests(request, call_next):
    """
    Log all incoming requests for monitoring.
    """
    start_time = datetime.now()
    
    # Process the request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = (datetime.now() - start_time).total_seconds()
    
    # Log request details
    logger.info(
        f"Request: {request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)
