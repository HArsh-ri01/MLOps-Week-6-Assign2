#!/bin/bash
# local_test.sh - Script to test the pipeline locally

set -e

echo "🧪 Running local tests for Iris ML Pipeline"

# Check if required files exist
echo "📋 Checking required files..."
required_files=("train_model.py" "validate_model.py" "generate_plots.py" "iris_fastapi.py" "requirements.txt" "Dockerfile")

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "❌ Missing required file: $file"
        exit 1
    fi
done
echo "✅ All required files present"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt
pip install pytest httpx

# Train model
echo "🏋️ Training model..."
python train_model.py

# Validate model
echo "✅ Validating model..."
python validate_model.py

# Generate plots
echo "📊 Generating plots..."
python generate_plots.py

# Test API
echo "🧪 Testing API..."
python -m pytest tests/test_api.py -v

# Test Docker build (optional)
if command -v docker &> /dev/null; then
    echo "🐳 Testing Docker build..."
    docker build -t iris-api-test .
    
    echo "🚀 Running Docker container test..."
    container_id=$(docker run -d -p 8201:8200 iris-api-test)
    
    # Wait for container to start
    sleep 10
    
    # Test the API
    echo "🔍 Testing containerized API..."
    curl -f http://localhost:8201/ || echo "Health check failed"
    curl -X POST "http://localhost:8201/predict/" \
        -H "Content-Type: application/json" \
        -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}' || echo "API test failed"
    
    # Clean up
    docker stop $container_id
    docker rm $container_id
    echo "🧹 Cleaned up test container"
else
    echo "⚠️ Docker not found, skipping container tests"
fi

echo "🎉 All local tests completed successfully!"
