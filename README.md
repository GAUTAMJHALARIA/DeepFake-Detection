## 🕵️‍♀️ Deepfake Detection (DFD)

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow Serving](https://img.shields.io/badge/TensorFlow%20Serving-2.14-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/tfx/guide/serving)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.12-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Poetry](https://img.shields.io/badge/Poetry-2.x-60A5FA?logo=poetry&logoColor=white)](https://python-poetry.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

An end‑to‑end, production‑ready deepfake detection system. It includes:

- **API**: FastAPI service for video upload and inference
- **Model Serving**: TensorFlow Serving hosting a SavedModel exported from your training notebooks
- **Preprocessing**: OpenCV Haar cascade face extraction + frame sampling, normalized to 64×64 RGB
- **Orchestration**: Docker Compose (local). Nginx, Celery/Redis, MinIO, and full observability are planned

✨ Highlights
- **Accurate backbone**: EfficientNetB0 with oversampling for class balance (see notebooks)
- **Simple deployment**: `docker compose up` and you’re live
- **Predict endpoint**: Upload videos and get per‑frame scores + aggregated decision
- **Configurable**: All knobs via environment variables

---

### Architecture

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

### Repository layout

- `api/app/main.py`: FastAPI app exposing `/health` and `/predict`
- `api/app/inference.py`: Frame extraction, face detection, preprocessing, TF‑Serving client, aggregation
- `settings.py`: Centralized configuration via environment variables
- `docker-compose.yml`: Two services: `tfserving` and `api`
- `Dockerfile`: Builds API image with Poetry and system deps (ffmpeg, OpenCV libs)
- `EfficientB0_OVERSAMPLING.ipynb`: Training with EfficientNetB0 + RandomOverSampler
- `EB0_OVS_PREDICTIONS.ipynb`: Export to SavedModel for TF‑Serving and batch predictions utilities
- `models/`: Place your exported SavedModel under `models/deepfake/<version>`

---

### How it works

- **Frame sampling**: Downsamples to target FPS (default 2.0) from the video
- **Face detection**: OpenCV Haar cascade; crops the largest face with margin and square crop fallback
- **Preprocess**: Convert BGR→RGB, resize to 64×64, normalize to [0,1]
- **Serving**: Send a batch `[N, 64, 64, 3]` to TensorFlow Serving `predict` endpoint
- **Aggregation**: Mean of frame scores → `label = "fake" if score ≥ THRESHOLD else "real"`

---

### Quickstart (Docker Compose)

1) Export your model to TensorFlow SavedModel format using the notebook (see below), then place it at:

```
models/deepfake/1/
  assets/
  variables/
  saved_model.pb
```

2) Start services:

```bash
docker compose up --build
```

This launches:
- `tfserving` on port `8501`
- `api` on port `8000`

3) Open API docs at `http://localhost:8000/docs` 🧪

---

### Demo 🎥

Embed a short GIF demo of the request/response or UI here. Place your media under `assets/` and reference it:

![Deepfake Detection Demo](assets/demo.gif)

If the image doesn't render on GitHub, ensure the file exists at `assets/demo.gif` and is committed to the repo.

---

### Screenshots 🖼️

Optional static screenshots:

<img src="assets/screenshot-swagger.png" width="800" alt="FastAPI Swagger UI" />

<img src="assets/screenshot-result.png" width="800" alt="Prediction Result" />

---

### Local development (without Docker)

```bash
cd api
poetry install --no-root
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Ensure TF‑Serving is running and reachable at `TF_SERVING_URL` (default `http://tfserving:8501`). For a local TF‑Serving container, mount your model:

```bash
docker run --rm -p 8501:8501 \
  -e MODEL_NAME=deepfake \
  -v ${PWD}/models/deepfake:/models/deepfake:ro \
  tensorflow/serving:2.14.1
```

---

### Exporting the model (from notebooks)

- Train: `EfficientB0_OVERSAMPLING.ipynb`
  - EfficientNetB0, mixed precision, oversampling via `RandomOverSampler`
  - Saves weights to `Models/Eb0_OVS_best_model_weights.h5` and full model to `Models/Eb0_OVS_best_model.h5`

- Export for serving: `EB0_OVS_PREDICTIONS.ipynb`
  - Rebuild the architecture and load weights
  - Export SavedModel, e.g.:

```python
import os
os.makedirs("models/deepfake/1", exist_ok=True)
model.save("models/deepfake/1", include_optimizer=False, save_format="tf")
```

Place that directory under the repo `models/` so `docker-compose.yml` mounts it into TF‑Serving.

---

### API

- Base URL: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

- Health
  - `GET /health`
  - Response example:

```json
{"ok": true, "tfserving": {"model_version_status": "..."}}
```

- Predict
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

### Configuration

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

### Development

- Linting & type checking (from repo root):

```bash
poetry run ruff check
poetry run mypy api
```

- Tests (placeholder):

```bash
poetry run pytest -q
```

---

### Roadmap 🔭

- Nginx reverse proxy + TLS (Let’s Encrypt)
- Background processing via Celery + Redis
- Object storage via MinIO for uploads and artifacts
- Observability: Prometheus + Grafana, ELK stack
- GPU‑accelerated face detection (e.g., RetinaFace) and tracking
- Model registry and automated rollouts

---

### Datasets and Citations 📚

Replace this list with the exact datasets you used. Common options in deepfake research:

- [Celeb-DF v2](https://arxiv.org/abs/1909.12962)
- [FaceForensics++](https://arxiv.org/abs/1901.08971)
- [DeepFake Detection Challenge (DFDC)](https://www.kaggle.com/c/deepfake-detection-challenge)
- [Google/Jigsaw DeepFakeDetection (DFD)](https://ai.googleblog.com/2019/09/contributing-data-to-deepfake-detection.html)

For each dataset, cite the paper and comply with the license/terms of use. Document your preprocessing steps, any filtering, and train/val/test splits for reproducibility.

---

### Troubleshooting 🛠️

- "No frames extracted": video has no detectable faces; fallback square‑crop path is included but may still fail on corrupted media
- TF‑Serving 404: verify your SavedModel path and `MODEL_NAME`
- CORS issues: CORS is permissive by default in `main.py` (`allow_origins=["*"]`)

---

### Acknowledgements

- EfficientNetB0 (ImageNet weights) via TensorFlow/Keras
- OpenCV Haar cascades for face detection
- Imbalanced‑learn for oversampling utilities

---

### License

License not specified. Add a `LICENSE` file to clarify permitted use.

---

### Maintainer

- **GAUTAM JHALARIA** · jhalariagautam@gmail.com
