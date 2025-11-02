#!/bin/bash

# Deepfake Detection System - Startup Script

echo "🚀 Starting Deepfake Detection System..."
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

# Start the services
echo "🔧 Starting services..."
docker compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service health..."

# Check TensorFlow Serving
if curl -s http://localhost:8501/v1/models/deepfake > /dev/null; then
    echo "✅ TensorFlow Serving is ready"
else
    echo "⚠️  TensorFlow Serving is not ready yet"
fi

# Check API
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API is ready"
else
    echo "⚠️  API is not ready yet"
fi

# Check Frontend
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Frontend is ready"
else
    echo "⚠️  Frontend is not ready yet"
fi

echo ""
echo "🎉 Deepfake Detection System is starting up!"
echo "======================================"
echo "📱 Web Dashboard: http://localhost:3000"
echo "🔧 API Documentation: http://localhost:8000/docs"
echo "📊 API Health: http://localhost:8000/health"
echo ""
echo "💡 Tips:"
echo "   - Use the web dashboard for the best experience"
echo "   - Upload videos or images via drag-and-drop"
echo "   - Try batch processing for multiple files"
echo "   - Analyze videos directly from URLs"
echo ""
echo "🛑 To stop: docker compose down"
echo "📋 To view logs: docker compose logs -f"
