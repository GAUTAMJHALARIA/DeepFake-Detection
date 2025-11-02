#!/bin/bash

echo "🚀 Starting FastAPI Server (Local Development)..."
echo "======================================"

cd api

# Set environment variable for local development
export TF_SERVING_URL=http://localhost:8501

echo "🔧 Starting FastAPI server with local TensorFlow Serving..."
echo "TF_SERVING_URL is set to: $TF_SERVING_URL"

poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
