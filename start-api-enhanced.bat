@echo off
echo 🚀 Starting Enhanced FastAPI Server...
echo ======================================

cd api

REM Set environment variables for enhanced features
set TF_SERVING_URL=http://localhost:8501
set REDIS_URL=redis://localhost:6379
set EXTRACT_ALL_FRAMES=true
set ENABLE_GRADCAM=true
set MAX_RESOLUTION=1920x1080
set FRAME_CACHE_TTL=3600
set CLEANUP_AFTER_ANALYSIS=true

echo 🔧 Starting Enhanced FastAPI server...
echo Environment:
echo   TF_SERVING_URL: %TF_SERVING_URL%
echo   REDIS_URL: %REDIS_URL%
echo   GRADCAM_ENABLED: %ENABLE_GRADCAM%
echo   MAX_RESOLUTION: %MAX_RESOLUTION%
echo.

poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
