@echo off
echo 🚀 Starting Deepfake Detection System...
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
echo 🔧 Starting services...
docker compose up -d

REM Wait for services to be ready
echo ⏳ Waiting for services to start...
timeout /t 10 /nobreak >nul

echo 🔍 Checking service health...

REM Check services (simplified for Windows)
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

curl -s http://localhost:3000 >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Frontend is not ready yet
) else (
    echo ✅ Frontend is ready
)

echo.
echo 🎉 Deepfake Detection System is starting up!
echo ======================================
echo 📱 Web Dashboard: http://localhost:3000
echo 🔧 API Documentation: http://localhost:8000/docs
echo 📊 API Health: http://localhost:8000/health
echo.
echo 💡 Tips:
echo    - Use the web dashboard for the best experience
echo    - Upload videos or images via drag-and-drop
echo    - Try batch processing for multiple files
echo    - Analyze videos directly from URLs
echo.
echo 🛑 To stop: docker compose down
echo 📋 To view logs: docker compose logs -f
pause
