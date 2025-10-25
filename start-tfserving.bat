@echo off
echo 🚀 Starting TensorFlow Serving...
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

REM Start TensorFlow Serving
echo 🔧 Starting TensorFlow Serving...
docker run --rm -d --name deepfake-tfserving ^
  -p 8501:8501 ^
  -e MODEL_NAME=deepfake ^
  -v "%cd%\models\deepfake:/models/deepfake:ro" ^
  tensorflow/serving:2.14.1

echo ⏳ Waiting for TensorFlow Serving to start...
timeout /t 10 /nobreak >nul

echo 🔍 Checking TensorFlow Serving health...
curl -s http://localhost:8501/v1/models/deepfake >nul 2>&1
if errorlevel 1 (
    echo ⚠️  TensorFlow Serving is not ready yet, please wait a moment...
) else (
    echo ✅ TensorFlow Serving is ready!
)

echo.
echo 🎉 TensorFlow Serving is running!
echo ======================================
echo 📊 Model Status: http://localhost:8501/v1/models/deepfake
echo 🛑 To stop: docker stop deepfake-tfserving
echo.
pause
