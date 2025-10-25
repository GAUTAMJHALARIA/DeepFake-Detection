@echo off
echo 🚀 Starting Memory-Optimized Deepfake Detection System...
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

REM Clear Redis cache first
echo 🧹 Clearing Redis cache...
redis-cli FLUSHALL >nul 2>&1

REM Set memory-optimized environment variables
set EXTRACT_ALL_FRAMES=false
set MAX_CACHED_FRAMES=50
set FRAME_CACHE_TTL=1800
set DEFAULT_FPS=1.0
set MAX_FRAMES=100

echo 🔧 Starting memory-optimized services...
echo    - Frame extraction: Sampled (not all frames)
echo    - Max cached frames: %MAX_CACHED_FRAMES%
echo    - Cache TTL: %FRAME_CACHE_TTL% seconds
echo    - Target FPS: %DEFAULT_FPS%

REM Start API with memory-optimized settings
cd api
start /B poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000

echo ⏳ Waiting for API to start...
timeout /t 10 /nobreak >nul

echo 🔍 Checking API health...
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo ⚠️  API is not ready yet
) else (
    echo ✅ Memory-optimized API is ready
)

echo.
echo 🎉 Memory-Optimized System is ready!
echo ======================================
echo 🌐 Web Interface: http://localhost:3000
echo 🔧 API Documentation: http://localhost:8000/docs
echo 📊 API Health: http://localhost:8000/health
echo.
echo 💡 Memory Optimizations Applied:
echo    ✅ Reduced frame caching (50 frames max)
echo    ✅ Shorter cache TTL (30 minutes)
echo    ✅ Lower target FPS (1.0 FPS)
echo    ✅ Frame sampling instead of all frames
echo.
echo 🌐 Starting frontend...
cd ..\frontend
start /B npm start
echo Frontend will be available at http://localhost:3000
echo.
echo 🛑 To stop: Ctrl+C in API terminal
echo 📋 To monitor Redis: redis-cli monitor
echo.
pause
