<!-- PROJECT LOGO -->
<p align="center">
  <img src="assets/logo.png" alt="DFD Logo" width="200" />`
</p>

<h1 align="center">🕵️‍♀️ Deepfake Detection (DFD)</h1>

<p align="center">
  <b>An end-to-end, production-ready deepfake detection system</b><br>
  <i>FastAPI · TensorFlow Serving · EfficientNetB0 · OpenCV · Docker</i>
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" /></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.116-009688?logo=fastapi&logoColor=white" /></a>
  <a href="https://www.tensorflow.org/tfx/guide/serving"><img src="https://img.shields.io/badge/TensorFlow%20Serving-2.14-FF6F00?logo=tensorflow&logoColor=white" /></a>
  <a href="https://opencv.org/"><img src="https://img.shields.io/badge/OpenCV-4.12-5C3EE8?logo=opencv&logoColor=white" /></a>
  <a href="https://python-poetry.org/"><img src="https://img.shields.io/badge/Poetry-2.x-60A5FA?logo=poetry&logoColor=white" /></a>
  <a href="https://www.docker.com/"><img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" /></a>
</p>

---

## 📑 Table of Contents
- [Overview](#overview)
- [Why Use DFD?](#why-use-dfd)
- [Features](#features)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Quickstart](#quickstart)
- [Demo](#demo)
- [Screenshots](#screenshots)
- [Model Training & Export](#model-training--export)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Development](#development)
- [Roadmap](#roadmap)
- [Datasets & Citations](#datasets--citations)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Community & Support](#community--support)
- [License](#license)
- [Maintainer](#maintainer)

---

## 📝 Overview
DFD is a robust, scalable deepfake detection system designed for research and production. It leverages state-of-the-art computer vision and deep learning techniques to detect manipulated videos with high accuracy.

---

## ❓ Why Use DFD?
- 🚀 **Fast & Scalable**: Built with FastAPI and TensorFlow Serving for high throughput.
- 🧠 **Modern ML Backbone**: EfficientNetB0, oversampling, and mixed precision for best results.
- 🛠️ **Easy Deployment**: Docker Compose orchestration, simple config.
- 🔒 **Secure**: Optional JWT authentication.
- 🧩 **Extensible**: Modular codebase, easy to add new models or preprocessing steps.

---

## ✨ Features
- Video upload and inference via REST API
- Per-frame face extraction and scoring
- Aggregated fake/real decision
- Configurable via environment variables
- Exportable models for TensorFlow Serving
- Demo and Swagger UI
- Dockerized for reproducibility

---

## 🏗️ Architecture
```mermaid
flowchart LR
  A[Client / SDK / cURL] -->|HTTP: /predict| B(FastAPI Service)
  B -->|frames, 64x64, RGB| C[Preprocessing\nOpenCV Haar + sampling]
  C -->|POST /v1/models/deepfake:predict| D[TensorFlow Serving]
  D -->|scores per frame| B
  B -->|JSON: {score, label, frame_samples}| A
  subgraph Runtime
    B
    C
    D
  end
```

---

## 📁 Repository Layout
- `api/app/main.py`: FastAPI app exposing `/health` and `/predict`
- `api/app/inference.py`: Frame extraction, face detection, preprocessing, TF‑Serving client, aggregation
- `settings.py`: Centralized configuration via environment variables
- `docker-compose.yml`: Two services: `tfserving` and `api`
- `Dockerfile`: Builds API image with Poetry and system deps (ffmpeg, OpenCV libs)
- `EfficientB0_OVERSAMPLING.ipynb`: Training with EfficientNetB0 + RandomOverSampler
- `EB0_OVS_PREDICTIONS.ipynb`: Export to SavedModel for TF‑Serving and batch predictions utilities
- `models/`: Place your exported SavedModel under `models/deepfake/<version>`

---

## 🚀 Quickstart
1. Export your model to TensorFlow SavedModel format using the notebook (see below), then place it at:
```
models/deepfake/1/
  assets/
  variables/
  saved_model.pb
```
2. Start services:
```bash
docker compose up --build
```
3. Open API docs at `http://localhost:8000/docs` 🧪

---

## 🎥 Demo
![Deepfake Detection Demo](assets/demo.gif)

---

## 🖼️ Screenshots
<img src="assets/screenshot-swagger.png" width="800" alt="FastAPI Swagger UI" />
<img src="assets/screenshot-result.png" width="800" alt="Prediction Result" />

---

## 📓 Model Training & Export
- **Train**: `EfficientB0_OVERSAMPLING.ipynb`
  - EfficientNetB0, mixed precision, oversampling via `RandomOverSampler`
  - Saves weights to `Models/Eb0_OVS_best_model_weights.h5` and full model to `Models/Eb0_OVS_best_model.h5`
- **Export for serving**: `EB0_OVS_PREDICTIONS.ipynb`
  - Rebuild the architecture and load weights
  - Export SavedModel, e.g.:
```python
import os
os.makedirs("models/deepfake/1", exist_ok=True)
model.save("models/deepfake/1", include_optimizer=False, save_format="tf")
```
Place that directory under the repo `models/` so `docker-compose.yml` mounts it into TF‑Serving.

---

## 🔌 API Reference
- **Base URL**: `http://localhost:8000`
- **Docs**: `http://localhost:8000/docs`
- **Health**
  - `GET /health`
  - Response example:
```json
{"ok": true, "tfserving": {"model_version_status": "..."}}
```
- **Predict**
  - `POST /predict`
  - Form-data: `file=@/path/to/video.mp4`
  - Query params: `fps` (optional, float)
  - Headers: `Authorization: Bearer <JWT_SECRET>` if auth enabled
Example request:
```bash
curl -X POST "http://localhost:8000/predict?fps=2.0" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./sample.mp4"
```
Response example:
```json
{
  "score": 0.7421,
  "label": "fake",
  "frame_samples": [
    {"t": 0.0, "score": 0.71},
    {"t": 0.5, "score": 0.78}
  ],
  "version": "1",
  "latency_ms": 1234,
  "meta": {
    "src_fps": 29.97,
    "total_frames": 1450,
    "used_step": 15,
    "face_frames": 128,
    "face_detect_rate": 0.9
  }
}
```

---

## ⚙️ Configuration
All settings are centralized in `settings.py` and can be set via environment variables (see `docker-compose.yml`).
- **Core model serving**
  - `MODEL_NAME` (default `deepfake`)
  - `MODEL_VERSION` (default `1`)
  - `TF_SERVING_URL` (default `http://tfserving:8501`)
- **Inference controls**
  - `DEFAULT_FPS` (default `2.0`)
  - `MAX_FRAMES` (default `256`)
  - `THRESHOLD` (default `0.5`)
  - `REQUEST_TIMEOUT` (default `30` seconds)
- **Auth**
  - `REQUIRE_AUTH` (default `false`)
  - `JWT_SECRET` (shared static token for demo purposes)
Create a `.env` at repo root if you prefer:
```
MODEL_NAME=deepfake
MODEL_VERSION=1
TF_SERVING_URL=http://tfserving:8501
DEFAULT_FPS=2.0
MAX_FRAMES=256
THRESHOLD=0.5
REQUEST_TIMEOUT=30
REQUIRE_AUTH=false
JWT_SECRET=change-me
```

---

## 🛠️ Development
- **Linting & type checking** (from repo root):
```bash
poetry run ruff check
poetry run mypy api
```
- **Tests** (placeholder):
```bash
poetry run pytest -q
```

---

## 🔭 Roadmap
- Nginx reverse proxy + TLS (Let’s Encrypt)
- Background processing via Celery + Redis
- Object storage via MinIO for uploads and artifacts
- Observability: Prometheus + Grafana, ELK stack
- GPU‑accelerated face detection (e.g., RetinaFace) and tracking
- Model registry and automated rollouts

---

## 📚 Datasets & Citations
Replace this list with the exact datasets you used. Common options in deepfake research:
- [Celeb-DF v2](https://arxiv.org/abs/1909.12962)
- [FaceForensics++](https://arxiv.org/abs/1901.08971)
- [DeepFake Detection Challenge (DFDC)](https://www.kaggle.com/c/deepfake-detection-challenge)
- [Google/Jigsaw DeepFakeDetection (DFD)](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html)
For each dataset, cite the paper and comply with the license/terms of use. Document your preprocessing steps, any filtering, and train/val/test splits for reproducibility.

---

## 🛠️ Troubleshooting
- "No frames extracted": video has no detectable faces; fallback square‑crop path is included but may still fail on corrupted media
- TF‑Serving 404: verify your SavedModel path and `MODEL_NAME`
- CORS issues: CORS is permissive by default in `main.py` (`allow_origins=["*"]`)

---

## ❓ FAQ
**Q: Can I use my own model?**
A: Yes! Replace the exported SavedModel in `models/deepfake/<version>` and update config if needed.

**Q: How do I enable authentication?**
A: Set `REQUIRE_AUTH=true` and provide a `JWT_SECRET` in your `.env`.

**Q: Can I run this on GPU?**
A: Yes, both training and serving support GPU acceleration if available.

---

## 🤝 Community & Support
- Issues and feature requests: [GitHub Issues](https://github.com/GAUTAMJHALARIA/DeepFake-Detection/issues)
- Email: jhalariagautam@gmail.com
- Pull requests welcome!

---

## 📄 License
License not specified. Add a `LICENSE` file to clarify permitted use.

---

## 👤 Maintainer
- **GAUTAM JHALARIA** · jhalariagautam@gmail.com
