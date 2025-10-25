#!/bin/bash

echo "🚀 Starting Enhanced Deepfake Detection System..."
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

# Start the enhanced services
echo "🔧 Starting enhanced services (Redis + TensorFlow Serving + API)..."
docker compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 20

echo "🔍 Checking service health..."

# Check Redis
if docker exec $(docker ps -q -f "name=redis") redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is ready"
else
    echo "⚠️  Redis is not ready yet"
fi

# Check TensorFlow Serving
if curl -s http://localhost:8501/v1/models/deepfake > /dev/null; then
    echo "✅ TensorFlow Serving is ready"
else
    echo "⚠️  TensorFlow Serving is not ready yet"
fi

# Check API
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Enhanced API is ready"
else
    echo "⚠️  Enhanced API is not ready yet"
fi

echo ""
echo "🎉 Enhanced Deepfake Detection System is ready!"
echo "======================================"
echo "🌐 Web Interface: http://localhost:3000"
echo "🔧 API Documentation: http://localhost:8000/docs"
echo "📊 API Health: http://localhost:8000/health"
echo "🧪 Enhanced Endpoints: http://localhost:8000/predict-enhanced"
echo ""
echo "💡 New Features Available:"
echo "   ✨ Frame-by-frame video analysis"
echo "   🎬 Interactive video player with confidence overlay"
echo "   🔥 Confidence heat maps and timeline visualization"
echo "   🧠 Grad-CAM++ explainable AI features"
echo "   📈 Advanced analytics dashboard"
echo "   🎯 Real-time processing with Redis caching"
echo ""
echo "🌐 Starting frontend development server..."
cd frontend
npm start &
echo "Frontend will be available at http://localhost:3000"
echo ""
echo "🛑 To stop: docker compose down"
echo "📋 To view logs: docker compose logs -f"
echo ""
