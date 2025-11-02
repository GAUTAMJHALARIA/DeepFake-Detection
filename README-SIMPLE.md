# Deepfake Detection System - Working Version

A functional deepfake detection system with FastAPI backend and React frontend.

## ✅ What's Working

- **FastAPI Backend** with video/image analysis endpoints
- **TensorFlow Serving** for model inference
- **React Frontend** with drag-and-drop file upload
- **Docker containerization** for easy deployment
- **Real-time analysis** with progress tracking

## 🚀 Quick Start

### Option 1: Full System (Recommended)
```bash
# Windows
start-simple.bat

# Linux/Mac
chmod +x start-simple.sh
./start-simple.sh
```

### Option 2: Manual Setup
```bash
# Start backend services
docker compose up -d

# Start frontend (in another terminal)
cd frontend
npm install
npm start
```

## 🌐 Access Points

- **Web Interface**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **API Health**: http://localhost:8000/health

## 🔧 API Endpoints

### Core Analysis
- `POST /predict` - Analyze video files
- `POST /predict-image` - Analyze image files
- `POST /predict-batch` - Batch processing
- `GET /health` - System health check
- `GET /supported-formats` - Supported file formats

### Example Usage
```bash
# Test with curl
curl -X POST "http://localhost:8000/predict" \
  -F "file=@your-video.mp4" \
  -H "Authorization: Bearer change-me"
```

## 📁 Project Structure

```
├── api/                    # FastAPI backend
│   ├── app/
│   │   ├── main.py        # API endpoints
│   │   └── inference.py   # ML processing
│   └── pyproject.toml     # Python dependencies
├── frontend/              # React frontend
│   ├── src/
│   │   ├── App.tsx        # Main app
│   │   └── components/    # UI components
│   └── package.json       # Node dependencies
├── models/deepfake/1/     # TensorFlow model
├── docker-compose.yml     # Container orchestration
└── Dockerfile            # API container
```

## 🎯 Features

### Web Interface
- **Drag & Drop Upload** - Easy file selection
- **Real-time Progress** - Live upload/processing status
- **Results Display** - Confidence scores and analysis
- **Multi-format Support** - Videos and images
- **Responsive Design** - Works on desktop and mobile

### API Features
- **Video Analysis** - Frame-by-frame deepfake detection
- **Image Analysis** - Single image processing
- **Batch Processing** - Multiple files at once
- **Face Detection** - Automatic face cropping
- **Configurable FPS** - Adjustable frame sampling
- **Health Monitoring** - System status checks

## ⚙️ Configuration

Environment variables (see `.env.example`):

```env
MODEL_NAME=deepfake
MODEL_VERSION=1
DEFAULT_FPS=2.0
MAX_FRAMES=256
THRESHOLD=0.5
REQUEST_TIMEOUT=30
REQUIRE_AUTH=false
JWT_SECRET=change-me
```

## 🛠️ Development

### Backend Development
```bash
cd api
poetry install
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
cd frontend
npm install
npm start
```

## 🐳 Docker Commands

```bash
# Start services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down

# Rebuild containers
docker compose up -d --build
```

## 🔍 Troubleshooting

### Common Issues

1. **Docker not running**
   - Start Docker Desktop
   - Verify with `docker info`

2. **Model not found**
   - Ensure `models/deepfake/1/` directory exists
   - Check model files are present

3. **Port conflicts**
   - Check if ports 8000, 8501, 3000 are available
   - Modify docker-compose.yml if needed

4. **Frontend build errors**
   - Run `npm install` in frontend directory
   - Clear node_modules and reinstall if needed

### Health Checks

```bash
# Check TensorFlow Serving
curl http://localhost:8501/v1/models/deepfake

# Check API
curl http://localhost:8000/health

# Check supported formats
curl http://localhost:8000/supported-formats
```

## 📊 Performance

- **Processing Speed**: ~2-5 seconds per video (depending on length)
- **Supported Formats**: MP4, AVI, MOV, MKV, WebM, JPG, PNG, etc.
- **Max File Size**: Limited by available memory
- **Concurrent Requests**: Supports multiple simultaneous analyses

## 🔒 Security

- Optional JWT authentication
- CORS enabled for web interface
- Input validation and sanitization
- Error handling and logging

## 🚧 Known Limitations

- Single model version (EfficientNet-B0)
- Basic face detection (Haar cascades)
- No persistent storage of results
- Limited to 64x64 input resolution

## 🎯 Future Enhancements

- Advanced analytics dashboard
- Historical analysis storage
- Multiple model support
- Enhanced face detection
- Real-time video streaming
- User management system

---

Built with ❤️ using FastAPI, React, TensorFlow Serving, and Docker.
