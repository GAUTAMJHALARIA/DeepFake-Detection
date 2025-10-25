@echo off
echo 🚀 Starting Deepfake Detection System (Basic Version)...
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

REM Start the services
echo 🔧 Starting backend services...
docker compose up -d

REM Wait for services to be ready
echo ⏳ Waiting for services to start...
timeout /t 15 /nobreak >nul

echo 🔍 Checking service health...

REM Check services
curl -s http://localhost:8501/v1/models/deepfake >nul 2>&1
if errorlevel 1 (
    echo ⚠️  TensorFlow Serving is not ready yet
) else (
    echo ✅ TensorFlow Serving is ready
)

curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo ⚠️  API is not ready yet
) else (
    echo ✅ API is ready
)

echo.
echo 🎉 Deepfake Detection System is ready!
echo ======================================
echo 🔧 API Documentation: http://localhost:8000/docs
echo 📊 API Health: http://localhost:8000/health
echo 🧪 Test API: http://localhost:8000/supported-formats
echo.
echo 💡 To test with a video file:
echo    curl -X POST "http://localhost:8000/predict" -F "file=@your-video.mp4" -H "Authorization: Bearer change-me"
echo.
echo 🛑 To stop: docker compose down
echo 📋 To view logs: docker compose logs -f
echo.
echo 🌐 Starting frontend development server...
cd frontend
start /B npm start
echo Frontend will be available at http://localhost:3000
echo.
pause
