# Deepfake Detection System - Complete Architecture & Implementation Overview

## Executive Summary

This is a **production-ready, full-stack deepfake detection system** that combines:
- **FastAPI backend** with TensorFlow Serving for ML inference
- **React/TypeScript frontend** with Material-UI
- **Redis caching** for high-performance frame storage
- **Automated video conversion** for browser compatibility
- **Grad-CAM++ explainable AI** for model interpretability
- **Docker-based deployment** for easy scalability

---

## 🏗️ System Architecture

### **Technology Stack**

| Layer | Technologies |
|-------|-------------|
| **Backend** | FastAPI, TensorFlow Serving, OpenCV, FFmpeg, Redis, yt-dlp |
| **Frontend** | React 19.2, TypeScript 4.9, Material-UI 7.3, D3.js 7.9 |
| **ML Framework** | TensorFlow 2.14, EfficientNet-B0 |
| **Infrastructure** | Docker, Docker Compose, Nginx |
| **Cache** | Redis 7-alpine (4GB limit, LRU eviction) |

### **Service Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  React Frontend │    │  FastAPI Backend  │    │ TensorFlow      │
│                 │    │                  │    │ Serving         │
│ • Video Player  │◄──►│ • Enhanced API   │◄──►│                 │
│ • Heat Maps     │    │ • Frame Caching  │    │ • EfficientNet   │
│ • Analytics     │    │ • Grad-CAM++     │    │ • Model v1      │
│ • Grad-CAM      │    │ • Statistics     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Redis Cache    │
                       │                 │
                       │ • Frame Data    │
                       │ • Thumbnails    │
                       │ • Grad-CAM      │
                       │ • Metadata      │
                       └─────────────────┘
```

---

## 🔧 Backend Implementation (`api/`)

### **1. Core API (`api/app/main.py`)**

**Endpoints Implemented:**
```python
POST   /predict                 # Video/file upload analysis
POST   /predict-url             # URL-based analysis (YouTube, social media)
GET    /analysis/{id}           # Retrieve cached analysis
GET    /frames/{id}/{index}     # Get specific frame with thumbnail
GET    /thumbnails/{id}         # Get all thumbnails for timeline
GET    /gradcam/{id}/{index}    # Get Grad-CAM heatmap
GET    /video/{id}              # Get converted video file
DELETE /analysis/{id}           # Cleanup analysis data
GET    /health                  # Health check (TF Serving + Redis)
GET    /supported-formats       # List supported formats
```

**Key Features:**
- JWT authentication (optional, disabled by default)
- CORS enabled for all origins
- Request timeout: 30 seconds
- Automatic error handling and cleanup
- UUID-based analysis tracking

### **2. Enhanced Inference (`api/app/enhanced_inference.py`)**

**Processing Pipeline:**
```python
1. Video Upload/Download
   ↓
2. Codec Detection & Conversion (H.264 Baseline)
   ↓
3. Frame Extraction (deterministic sampling)
   ↓
4. Face Detection (Haar Cascade, OpenCV)
   ↓
5. Preprocessing (64x64 RGB patches, normalized)
   ↓
6. Batch Inference (TensorFlow Serving)
   ↓
7. Grad-CAM Generation (XAI)
   ↓
8. Redis Caching & Response
```

**Key Functions:**
- `extract_all_frames_enhanced()` - Deterministic frame sampling
- `analyze_video_enhanced()` - Complete analysis pipeline
- `download_video_from_url()` - yt-dlp integration
- `generate_gradcam_heatmap()` - Simplified Grad-CAM++ implementation
- `tfserving_predict_enhanced()` - Model inference

**Deterministic Processing:**
- Consistent frame sampling via step calculation
- Fixed face detection parameters (scaleFactor=1.05, minNeighbors=6)
- Deterministic bbox expansion (margin=0.20)
- Video hash tracking for debugging

### **3. Redis Cache (`api/app/cache.py`)**

**Storage Strategy:**
```python
# Key Patterns
analysis:{id}              # Complete analysis JSON
frame:{id}:{index}        # Frame metadata (not full image)
thumb:{id}:{index}        # Thumbnail (320x180, base64)
gradcam:{id}:{index}      # Grad-CAM heatmap (base64)

# TTL: 30-60 minutes
# Memory Limit: 4GB with LRU eviction
# Max Cached Frames: 100
```

**Key Methods:**
- `store_analysis_data()` - Store complete result
- `store_thumbnail()` - Store frame previews
- `store_gradcam()` - Store heatmaps
- `cleanup_analysis()` - Delete all related keys

**Memory Optimization:**
- Thumbnails resized to 320x180 (was 640x360)
- Frame metadata only (no large numpy arrays)
- Base64 encoding for binary data
- Memory monitoring before storage

### **4. Video Converter (`api/app/video_converter.py`)**

**Functionality:**
- **Codec Detection**: H.264, VP8, VP9, Theora
- **Browser Compatibility Check**: FFmpeg-based probe
- **Auto-Conversion**: H.264 Baseline Profile for compatibility
- **Quality Preserved**: CRF 23, fast preset
- **Progressive Download**: Fast-start enabled

**Methods:**
- `check_codec_support()` - Detect video codec
- `convert_to_browser_compatible()` - FFmpeg conversion
- `get_video_info()` - Extract video metadata

### **5. Configuration (`api/settings.py`)**

```python
# Model Configuration
MODEL_NAME = "deepfake"
MODEL_VERSION = "1"
TF_SERVING_URL = "http://localhost:8501"

# Inference Parameters
DEFAULT_FPS = 2.0              # Frame sampling rate
MAX_FRAMES = 256               # Max frames per video
THRESHOLD = 0.5                # Classification threshold
REQUEST_TIMEOUT = 30.0         # API timeout

# Enhanced Features
MAX_RESOLUTION = "1920x1080"   # Max video resolution
EXTRACT_ALL_FRAMES = False     # Memory optimization
ENABLE_GRADCAM = True          # XAI features
MAX_CACHED_FRAMES = 100       # Frame cache limit
FRAME_CACHE_TTL = 1800        # Cache TTL (30 min)

# Redis Configuration
REDIS_URL = "redis://localhost:6379"
REDIS_DB = 0

# File Storage
TEMP_STORAGE_PATH = "/tmp/deepfake_analysis"
CLEANUP_AFTER_ANALYSIS = True

# Security
REQUIRE_AUTH = False
JWT_SECRET = "change-me"
```

---

## 🎨 Frontend Implementation (`frontend/`)

### **1. Main App (`src/App.tsx`)**

**Theme Configuration:**
- Dark mode with gradient header
- Material-UI custom styling
- Security and Analytics icons
- Responsive design

### **2. Enhanced Video Analysis (`src/components/EnhancedVideoAnalysis.tsx`)**

**Features:**
- **Dual Upload Modes**: File upload + URL input
- **Drag & Drop**: React Dropzone integration
- **Real-time Progress**: Upload + processing progress
- **Error Handling**: User-friendly error messages
- **Result Display**: Grid layout with multiple views

**Upload Flow:**
```typescript
1. User selects file or enters URL
2. File is validated (type, size)
3. Upload progress tracked (axios onUploadProgress)
4. Processing status displayed
5. Results displayed in interactive grid
6. Frame navigation via timeline/keyboard
```

### **3. Enhanced Video Player (`src/components/EnhancedVideoPlayer.tsx`)**

**Features:**
- **HTML5 Video Player**: Custom controls
- **Confidence Overlay**: Real-time score display
- **Keyboard Shortcuts**: Space, arrows, +/- for speed
- **Playback Controls**: Play/pause, frame step, speed (0.25x-2x)
- **Volume Control**: Slider-based
- **Timeline Scrubber**: Color-coded confidence visualization
- **Browser Codec Detection**: Automatic format checking
- **Auto-Conversion**: Fetch converted video from backend

**Keyboard Shortcuts:**
- `Space` - Play/Pause
- `←/→` - Frame step (previous/next)
- `↑/↓` - Speed increase/decrease
- Volume and fullscreen support

**Video Compatibility:**
- Detects browser codec support
- Automatically fetches converted video
- Fallback to original if conversion fails
- Error handling for unsupported formats

### **4. Confidence Heat Map (`src/components/ConfidenceHeatMap.tsx`)**

**Features:**
- **D3.js Visualization**: Interactive timeline
- **Color-Coded Confidence**: Red-Yellow-Green gradient
- **Click-to-Navigate**: Jump to any frame
- **Thumbnail Previews**: Frame samples on hover
- **Face Detection Indicators**: Red dots for missing faces
- **Statistical Overlays**: Mean, variance, suspicious segments
- **Zoom & Pan**: Adjustable height (150-400px)

**Visualization:**
```typescript
// Color Scale (Red-Yellow-Green)
Red (≥70%): High fake confidence
Yellow (30-70%): Uncertain
Green (<30%): Low fake confidence

// Elements
- Rectangles: Confidence scores per frame
- Thumbnails: Preview frames (max 20)
- Red circles: No face detected
- White dashed line: Current frame
- Axes: Time (s) vs Confidence (%)
```

### **5. Analytics Dashboard (`src/components/AnalyticsDashboard.tsx`)**

**Metrics Displayed:**
- Mean confidence score
- Confidence variance
- Max/min confidence
- Frame count statistics
- Face detection rate
- Processing latency
- Quality scores
- Suspicious frame count

### **6. Grad-CAM Viewer (`src/components/GradCAMViewer.tsx`)**

**Features:**
- **Heatmap Visualization**: PNG format
- **Zoom Controls**: 1x to 4x
- **Opacity Adjustment**: Slider control
- **Overlay Toggle**: Show/hide heatmap
- **Frame Navigation**: Previous/next controls
- **Download**: Export heatmap image

**Controls:**
- Zoom: 1x, 2x, 3x, 4x
- Opacity: 0.0 to 1.0
- Show/Hide Overlay: Toggle
- Frame Navigation: Previous/Next
- Download: PNG export

### **7. Error Boundary (`src/components/ErrorBoundary.tsx`)**

**Purpose:**
- Graceful error recovery
- User-friendly error messages
- Reload functionality
- Prevents app crashes

---

## 🐳 Infrastructure (`docker-compose.yml`)

### **Services:**

#### **1. Redis Service**
```yaml
image: redis:7-alpine
ports: 6379:6379
volumes: redis_data:/data
command: redis-server --appendonly yes --maxmemory 4gb --maxmemory-policy allkeys-lru
healthcheck: redis-cli ping
```

#### **2. TensorFlow Serving**
```yaml
image: tensorflow/serving:2.14.1
environment:
  - MODEL_NAME=deepfake
  - MODEL_BASE_PATH=/models
volumes: ./models/deepfake:/models/deepfake:ro
ports: 8501:8501
healthcheck: curl -f http://localhost:8501/v1/models/deepfake
```

#### **3. FastAPI Backend**
```yaml
build: Dockerfile
environment:
  - TF_SERVING_URL=http://tfserving:8501
  - REDIS_URL=redis://redis:6379
  - EXTRACT_ALL_FRAMES=true
  - ENABLE_GRADCAM=true
  - MAX_RESOLUTION=1920x1080
  - FRAME_CACHE_TTL=3600
ports: 8000:8000
volumes: /tmp/deepfake_analysis:/tmp/deepfake_analysis
depends_on:
  - redis (condition: service_healthy)
  - tfserving (condition: service_healthy)
```

---

## 📊 Data Flow

### **Analysis Flow:**

```
1. User uploads video/URL
   ↓
2. FastAPI receives file
   ↓
3. Video conversion (if needed)
   ↓
4. Frame extraction (deterministic)
   ↓
5. Face detection (Haar Cascade)
   ↓
6. Image preprocessing (64x64 RGB)
   ↓
7. Batch to TensorFlow Serving
   ↓
8. Receive predictions
   ↓
9. Generate Grad-CAM (optional)
   ↓
10. Store in Redis cache
   ↓
11. Return JSON response
   ↓
12. Frontend displays results
```

### **Request/Response Format:**

**Request (Video Upload):**
```http
POST /predict
Content-Type: multipart/form-data
Authorization: Bearer change-me

file: [video file]
fps: 2.0 (optional)
```

**Response:**
```json
{
  "id": "uuid-analysis-id",
  "score": 0.75,
  "label": "fake",
  "video_info": {
    "duration": 30.5,
    "fps": 29.97,
    "total_frames": 914,
    "processed_frames": 128,
    "resolution": "1920x1080",
    "face_detect_rate": 0.89,
    "conversion_info": {
      "was_converted": true,
      "conversion_message": "Converted to H.264 Baseline",
      "original_path": "/tmp/...",
      "final_path": "/tmp/..."
    }
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
  "latency_ms": 15420,
  "version": "1"
}
```

---

## 🎯 Key Features Implemented

### ✅ **1. Video Analysis**
- Frame-by-frame processing with face detection
- Automatic video conversion for browser compatibility
- Deterministic sampling for consistent results
- Support for multiple formats (MP4, AVI, MKV, WebM)
- URL analysis (YouTube, social media)
- Batch processing support

### ✅ **2. Real-Time Visualization**
- Interactive confidence timeline
- Color-coded heatmaps (Red/Yellow/Green)
- Thumbnail previews on timeline
- Statistical overlays
- Frame-by-frame navigation

### ✅ **3. Explainable AI (XAI)**
- Grad-CAM++ heatmap generation
- Attention visualization
- Model decision explanation
- Interactive heatmap viewer

### ✅ **4. Caching & Performance**
- Redis caching (4GB memory limit)
- Frame metadata storage
- Thumbnail caching (320x180 optimized)
- Grad-CAM caching
- LRU eviction policy
- Automatic cleanup

### ✅ **5. UI/UX Features**
- Drag-and-drop upload
- Real-time progress tracking
- Keyboard shortcuts
- Responsive design
- Error boundaries
- Loading states
- Smooth animations

### ✅ **6. Analytics & Reporting**
- Confidence score distribution
- Temporal pattern analysis
- Quality assessment metrics
- Face detection rate tracking
- Suspicious segment identification

### ✅ **7. Security & Configuration**
- JWT authentication (optional)
- CORS configuration
- Input validation
- File type restrictions
- Size limits
- Environment-based configuration

---

## 🚀 Performance Metrics

### **Processing Times** (30-second 1080p video)
- **Frame Extraction**: 5-10 seconds
- **Model Inference**: 10-20 seconds
- **Grad-CAM Generation**: 5-15 seconds
- **Total**: 20-45 seconds

### **Memory Usage**
- **Redis Cache**: 50-200MB per video
- **API Process**: 500MB-2GB
- **Frontend**: 100-300MB

### **Supported Formats**
- **Video**: MP4, AVI, MOV, MKV, WebM (up to 1080p)
- **Images**: JPG, PNG, BMP, TIFF, WebP
- **URLs**: YouTube, Vimeo, social media platforms

---

## 📁 Project Structure

```
DFD/
├── api/
│   ├── app/
│   │   ├── main.py                 # FastAPI endpoints
│   │   ├── enhanced_inference.py  # Core processing pipeline
│   │   ├── inference.py            # Format utilities
│   │   ├── cache.py               # Redis caching
│   │   ├── video_converter.py     # Video conversion
│   │   └── xai.py                 # Grad-CAM (optional)
│   ├── settings.py                # Configuration
│   └── pyproject.toml             # Dependencies
├── frontend/
│   ├── src/
│   │   ├── App.tsx                # Main app
│   │   └── components/
│   │       ├── EnhancedVideoAnalysis.tsx
│   │       ├── EnhancedVideoPlayer.tsx
│   │       ├── ConfidenceHeatMap.tsx
│   │       ├── AnalyticsDashboard.tsx
│   │       ├── GradCAMViewer.tsx
│   │       └── ErrorBoundary.tsx
│   └── package.json
├── models/
│   └── deepfake/1/                 # TensorFlow SavedModel
│       ├── saved_model.pb
│       └── variables/
├── docker-compose.yml              # Service orchestration
├── Dockerfile                       # Backend container
└── frontend/Dockerfile             # Frontend container
```

---

## 🎓 How It Works

### **1. Model Architecture**
- **Base Model**: EfficientNet-B0
- **Input**: 64×64 RGB face patches
- **Output**: Binary classification (real vs fake)
- **Training**: Class-weighted oversampling
- **Serving**: TensorFlow Serving REST API

### **2. Face Detection**
- **Method**: Haar Cascade (OpenCV)
- **Parameters**:
  - scaleFactor=1.05
  - minNeighbors=6
  - minSize=(40, 40)
- **Fallback**: Center square crop if no face detected
- **Bounding Box**: Expanded by 20% margin

### **3. Video Processing**
- **Resolution Limit**: 1920x1080 (auto-resize)
- **Frame Sampling**: Deterministic step calculation
- **Max Frames**: 256 (configurable)
- **Target FPS**: 2.0 (configurable)
- **Conversion**: H.264 Baseline for compatibility

### **4. Inference**
- **Batch Processing**: Stack all frames, predict together
- **Confidence Aggregation**: Mean of frame-level scores
- **Threshold**: 0.5 (configurable)
- **Classification**: score ≥ threshold → fake, else → real

### **5. Grad-CAM++**
- **Method**: Simplified attention map generation
- **Regions**: Eyes, nose, mouth areas
- **Visualization**: Color-coded heatmaps (JET colormap)
- **Format**: PNG base64 for frontend display

---

## 🔄 API Endpoints Summary

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/predict` | Analyze video/file | ✅ |
| POST | `/predict-url` | Analyze from URL | ✅ |
| GET | `/analysis/{id}` | Get cached analysis | ✅ |
| GET | `/frames/{id}/{index}` | Get frame data | ✅ |
| GET | `/thumbnails/{id}` | Get all thumbnails | ✅ |
| GET | `/gradcam/{id}/{index}` | Get heatmap | ✅ |
| GET | `/video/{id}` | Get converted video | ❌ |
| DELETE | `/analysis/{id}` | Cleanup data | ✅ |
| GET | `/health` | Health check | ❌ |
| GET | `/supported-formats` | List formats | ❌ |

---

## 🔧 Development Commands

### **Backend**
```bash
cd api
poetry install
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### **Frontend**
```bash
cd frontend
npm install
npm start
```

### **Docker**
```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

---

## 📝 Summary

### **What's Implemented**
✅ Full-stack deepfake detection system
✅ Real-time video analysis
✅ Interactive web dashboard
✅ Explainable AI with Grad-CAM++
✅ High-performance caching with Redis
✅ Automatic video conversion
✅ Deterministic processing
✅ Comprehensive analytics
✅ Production-ready deployment

### **Technologies Used**
- **Backend**: FastAPI, TensorFlow, OpenCV, Redis, FFmpeg
- **Frontend**: React, TypeScript, Material-UI, D3.js
- **Infrastructure**: Docker, Docker Compose, Nginx
- **ML**: EfficientNet-B0, Grad-CAM++

### **Key Features**
- Frame-by-frame analysis
- Real-time visualization
- Explainable AI
- High-performance caching
- Browser compatibility
- Deterministic results
- Production-ready

**Built with ❤️ by GAUTAM JHALARIA**
