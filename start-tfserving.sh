#!/bin/bash

echo "🚀 Starting TensorFlow Serving..."
echo "======================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if models exist
if [ ! -d "models/deepfake/1" ]; then
    echo "❌ Model directory not found. Please ensure models/deepfake/1/ exists."
    exit 1
fi

# Start TensorFlow Serving
echo "🔧 Starting TensorFlow Serving..."
docker run --rm -d --name deepfake-tfserving \
  -p 8501:8501 \
  -e MODEL_NAME=deepfake \
  -v "$(pwd)/models/deepfake:/models/deepfake:ro" \
  tensorflow/serving:2.14.1

echo "⏳ Waiting for TensorFlow Serving to start..."
sleep 10

echo "🔍 Checking TensorFlow Serving health..."
if curl -s http://localhost:8501/v1/models/deepfake > /dev/null; then
    echo "✅ TensorFlow Serving is ready!"
else
    echo "⚠️  TensorFlow Serving is not ready yet, please wait a moment..."
fi

echo ""
echo "🎉 TensorFlow Serving is running!"
echo "======================================"
echo "📊 Model Status: http://localhost:8501/v1/models/deepfake"
echo "🛑 To stop: docker stop deepfake-tfserving"
echo ""
