@echo off
echo 🚀 Starting Enhanced Deepfake Detection System...
echo ======================================

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker first.
    pause
    exit /b 1
)

REM Check if models exist
if not exist "models\deepfake\1" (
    echo ❌ Model directory not found. Please ensure models\deepfake\1\ exists.
    pause
    exit /b 1
)

REM Start the enhanced services
echo 🔧 Starting enhanced services (Redis + TensorFlow Serving + API)...
docker compose up -d

REM Wait for services to be ready
echo ⏳ Waiting for services to start...
timeout /t 20 /nobreak >nul

echo 🔍 Checking service health...

REM Check Redis
docker exec -it $(docker ps -q -f "name=redis") redis-cli ping >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Redis is not ready yet
) else (
    echo ✅ Redis is ready
)

REM Check TensorFlow Serving
curl -s http://localhost:8501/v1/models/deepfake >nul 2>&1
if errorlevel 1 (
    echo ⚠️  TensorFlow Serving is not ready yet
) else (
    echo ✅ TensorFlow Serving is ready
)

REM Check API
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Enhanced API is not ready yet
) else (
    echo ✅ Enhanced API is ready
)

echo.
echo 🎉 Enhanced Deepfake Detection System is ready!
echo ======================================
echo 🌐 Web Interface: http://localhost:3000
echo 🔧 API Documentation: http://localhost:8000/docs
echo 📊 API Health: http://localhost:8000/health
echo 🧪 Enhanced Endpoints: http://localhost:8000/predict-enhanced
echo.
echo 💡 New Features Available:
echo    ✨ Frame-by-frame video analysis
echo    🎬 Interactive video player with confidence overlay
echo    🔥 Confidence heat maps and timeline visualization
echo    🧠 Grad-CAM++ explainable AI features
echo    📈 Advanced analytics dashboard
echo    🎯 Real-time processing with Redis caching
echo.
echo 🌐 Starting frontend development server...
cd frontend
start /B npm start
echo Frontend will be available at http://localhost:3000
echo.
echo 🛑 To stop: docker compose down
echo 📋 To view logs: docker compose logs -f
echo.
pause
