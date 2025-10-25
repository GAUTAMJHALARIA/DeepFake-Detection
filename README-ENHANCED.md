# 🚀 Enhanced Deepfake Detection System

A **production-ready, full-stack deepfake detection system** with advanced video analysis, real-time visualization, and explainable AI features.

## ✨ Enhanced Features

### 🎬 **Advanced Video Analysis**
- **Frame-by-Frame Processing** - Extract and analyze every frame up to 1080p
- **Interactive Video Player** - Custom player with confidence overlay and timeline
- **Real-time Confidence Visualization** - Live confidence scores during playback
- **Keyboard Shortcuts** - Space (play/pause), arrows (frame step), up/down (speed)

### 🔥 **Confidence Heat Maps**
- **Interactive Timeline** - Click anywhere to jump to that frame
- **Color-coded Visualization** - Red-Yellow-Green confidence indicators
- **Temporal Pattern Analysis** - Identify suspicious segments over time
- **Statistical Overlays** - Face detection issues and quality metrics

### 🧠 **Explainable AI (Grad-CAM++)**
- **Visual Explanations** - See which pixels influence model decisions
- **Attention Heatmaps** - Red areas show highest model attention
- **Interactive Controls** - Adjust opacity, zoom, and overlay settings
- **Downloadable Results** - Export heatmaps for further analysis

### 📊 **Advanced Analytics Dashboard**
- **Comprehensive Statistics** - Mean, variance, distribution analysis
- **Quality Assessment** - Video quality and face detection metrics
- **Temporal Trends** - Confidence patterns over time
- **Suspicious Segments** - Automatically detected high-risk areas

### ⚡ **Performance & Caching**
- **Redis Integration** - Fast frame caching and retrieval
- **Background Processing** - Non-blocking analysis pipeline
- **Memory Optimization** - Efficient frame storage and cleanup
- **Scalable Architecture** - Ready for production deployment

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React Frontend │    │   FastAPI Backend │    │ TensorFlow      │
│                 │    │                  │    │ Serving         │
│ • Video Player  │◄──►│ • Enhanced API   │◄──►│                 │
│ • Heat Maps     │    │ • Frame Caching  │    │ • EfficientNet  │
│ • Analytics     │    │ • Grad-CAM++     │    │ • Model v1      │
│ • Grad-CAM      │    │ • Statistics     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Redis Cache     │
                       │                 │
                       │ • Frame Data    │
                       │ • Thumbnails    │
                       │ • Grad-CAM      │
                       │ • Metadata      │
                       └─────────────────┘
```

## 🚀 Quick Start

### Option 1: Enhanced System (Recommended)
```bash
# Windows
start-enhanced.bat

# Linux/Mac
chmod +x start-enhanced.sh
./start-enhanced.sh
```

### Option 2: Docker Compose
```bash
# Start all services
docker compose up -d

# Start frontend separately
cd frontend
npm install
npm start
```

## 🌐 Access Points

- **🎬 Web Interface**: http://localhost:3000
- **📚 API Documentation**: http://localhost:8000/docs
- **💓 Health Check**: http://localhost:8000/health
- **🧪 Enhanced Analysis**: http://localhost:8000/predict-enhanced

## 🔧 Enhanced API Endpoints

### **Core Analysis**
```http
POST /predict-enhanced
Content-Type: multipart/form-data
Authorization: Bearer change-me

# Enhanced analysis with full feature set
```

### **Frame Management**
```http
GET /frames/{analysis_id}/{frame_index}     # Get specific frame data
GET /thumbnails/{analysis_id}               # Get all thumbnails
GET /analysis/{analysis_id}                 # Get cached analysis
DELETE /analysis/{analysis_id}              # Cleanup analysis data
```

### **Explainable AI**
```http
GET /gradcam/{analysis_id}/{frame_index}    # Get Grad-CAM heatmap
```

### **URL Analysis**
```http
POST /predict-url-enhanced
Content-Type: application/json
{
  "url": "https://youtube.com/watch?v=..."
}
```

## 📊 Response Format

### Enhanced Analysis Response
```json
{
  "id": "analysis-uuid",
  "score": 0.75,
  "label": "fake",
  "video_info": {
    "duration": 30.5,
    "fps": 29.97,
    "resolution": "1920x1080",
    "processed_frames": 914,
    "face_detect_rate": 0.89
  },
  "frames": [
    {
      "index": 0,
      "timestamp": 0.0,
      "confidence": 0.65,
      "label": "fake",
      "face_detected": true,
      "confidence_color": [255, 165, 0],
      "has_gradcam": true
    }
  ],
  "statistics": {
    "mean_confidence": 0.72,
    "confidence_variance": 0.15,
    "max_confidence": 0.95,
    "min_confidence": 0.45,
    "suspicious_frames": 45,
    "quality_score": 0.87
  },
  "processing_info": {
    "gradcam_enabled": true,
    "all_frames_extracted": true,
    "max_resolution": "1920x1080",
    "threshold": 0.5
  },
  "latency_ms": 15420
}
```

## 🎯 Usage Examples

### **Web Interface**
1. **Upload Video** - Drag & drop or click to select
2. **Choose Analysis Type** - Basic or Enhanced
3. **Watch Processing** - Real-time progress updates
4. **Explore Results** - Interactive tabs for different views
5. **Navigate Frames** - Click timeline or use keyboard shortcuts

### **API Usage**
```bash
# Enhanced video analysis
curl -X POST "http://localhost:8000/predict-enhanced" \
  -F "file=@video.mp4" \
  -H "Authorization: Bearer change-me"

# Get specific frame
curl "http://localhost:8000/frames/analysis-id/42" \
  -H "Authorization: Bearer change-me"

# Get Grad-CAM heatmap
curl "http://localhost:8000/gradcam/analysis-id/42" \
  -H "Authorization: Bearer change-me"
```

## ⚙️ Configuration

### Environment Variables
```env
# Enhanced Features
EXTRACT_ALL_FRAMES=true
ENABLE_GRADCAM=true
MAX_RESOLUTION=1920x1080
FRAME_CACHE_TTL=3600

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_DB=0

# Processing Limits
DEFAULT_FPS=2.0
MAX_FRAMES=256
THRESHOLD=0.5
REQUEST_TIMEOUT=30

# Storage
TEMP_STORAGE_PATH=/tmp/deepfake_analysis
CLEANUP_AFTER_ANALYSIS=true
```

## 🎨 Frontend Components

### **Enhanced Video Player**
- Custom HTML5 video player with confidence overlay
- Synchronized timeline with color-coded confidence
- Frame-by-frame navigation with keyboard shortcuts
- Playback speed control (0.25x to 2x)
- Volume control and fullscreen support

### **Confidence Heat Map**
- D3.js-powered interactive visualization
- Click-to-navigate timeline scrubbing
- Statistical overlays and quality indicators
- Zoom and pan capabilities
- Export functionality

### **Analytics Dashboard**
- Comprehensive statistical analysis
- Quality assessment metrics
- Temporal pattern detection
- Suspicious segment identification
- Processing performance metrics

### **Grad-CAM Viewer**
- Visual explanation of model decisions
- Interactive heatmap overlays
- Opacity and zoom controls
- Frame navigation and comparison
- Technical interpretation guides

## 🔧 Development

### **Backend Development**
```bash
cd api
poetry install
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### **Frontend Development**
```bash
cd frontend
npm install
npm start
```

### **Redis Development**
```bash
# Start Redis locally
redis-server

# Monitor Redis
redis-cli monitor
```

## 🐳 Docker Commands

```bash
# Start enhanced system
docker compose up -d

# View logs
docker compose logs -f api
docker compose logs -f redis

# Scale services
docker compose up -d --scale api=2

# Cleanup
docker compose down -v
```

## 📈 Performance Metrics

### **Processing Times** (1080p, 30-second video)
- **Frame Extraction**: ~5-10 seconds
- **ML Inference**: ~10-20 seconds
- **Grad-CAM Generation**: ~5-15 seconds
- **Total Processing**: ~20-45 seconds

### **Memory Usage**
- **Redis Cache**: ~50-200MB per video
- **API Process**: ~500MB-2GB
- **Frontend**: ~100-300MB

### **Supported Formats**
- **Video**: MP4, AVI, MOV, MKV, WebM (up to 1080p)
- **Images**: JPG, PNG, BMP, TIFF, WebP
- **URLs**: YouTube, Vimeo, social media platforms

## 🔒 Security Features

- **JWT Authentication** - Optional API security
- **Input Validation** - File type and size limits
- **CORS Protection** - Configurable origin policies
- **Data Cleanup** - Automatic temporary file removal
- **Rate Limiting** - Configurable request limits

## 🚧 Troubleshooting

### **Common Issues**

1. **Redis Connection Failed**
   ```bash
   # Check Redis status
   docker ps | grep redis
   docker logs redis-container-name
   ```

2. **Out of Memory**
   ```bash
   # Increase Docker memory limit
   # Clear Redis cache
   redis-cli FLUSHALL
   ```

3. **Slow Processing**
   ```bash
   # Reduce frame extraction
   export EXTRACT_ALL_FRAMES=false
   export DEFAULT_FPS=1.0
   ```

4. **Frontend Build Errors**
   ```bash
   cd frontend
   rm -rf node_modules package-lock.json
   npm install
   ```

### **Health Checks**
```bash
# Check all services
curl http://localhost:8000/health
curl http://localhost:8501/v1/models/deepfake
redis-cli ping

# Check Redis data
redis-cli keys "analysis:*"
redis-cli keys "frame:*"
```

## 🎯 Future Enhancements

- **Real-time Streaming** - Live camera analysis
- **Multi-model Ensemble** - Combine multiple detection models
- **Advanced Preprocessing** - Face alignment and enhancement
- **Mobile Support** - Responsive design and touch controls
- **Collaborative Features** - Team workspaces and sharing
- **Advanced Export** - PDF reports and annotated videos

## 📊 System Requirements

### **Minimum**
- **CPU**: 4 cores, 2.5GHz
- **RAM**: 8GB
- **Storage**: 10GB free space
- **Docker**: 20.10+
- **Node.js**: 18+

### **Recommended**
- **CPU**: 8 cores, 3.0GHz
- **RAM**: 16GB
- **GPU**: NVIDIA GPU with CUDA support
- **Storage**: 50GB SSD
- **Network**: High-speed internet for URL analysis

---

## 🏆 **Enhanced Edition Features Summary**

✅ **Interactive Video Player** with confidence overlay
✅ **Real-time Heat Maps** with click navigation
✅ **Grad-CAM++ Explainability** with visual attention
✅ **Advanced Analytics** with statistical insights
✅ **Redis Caching** for fast frame retrieval
✅ **Frame-by-Frame Analysis** with full metadata
✅ **URL Analysis** from social media platforms
✅ **Mobile-Responsive** design for all devices
✅ **Export Capabilities** for data and visualizations
✅ **Production-Ready** with health checks and monitoring

**Built with ❤️ using FastAPI, React, TensorFlow, Redis, and Docker**
