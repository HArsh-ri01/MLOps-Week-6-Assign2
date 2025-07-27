#!/bin/bash
# deploy_local.sh - Script to deploy locally for testing

set -e

echo "🚀 Deploying Iris API locally"

# Check if model exists
if [[ ! -f "model.joblib" ]]; then
    echo "📚 Model not found, training..."
    python train_model.py
fi

# Start the API
echo "🌟 Starting FastAPI server..."
echo "API will be available at: http://localhost:8200"
echo "Press Ctrl+C to stop"

uvicorn iris_fastapi:app --host 0.0.0.0 --port 8200 --reload
