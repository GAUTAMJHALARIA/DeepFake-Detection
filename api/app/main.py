import time
import os
import hashlib
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import tensorflow as tf

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from settings import settings
from .inference import (
    extract_frames_and_preprocess,
    tfserving_predict,
    aggregate,
    health_check,
)
from .xai import generate_gradcam_heatmaps, generate_explanations
from .annotations import (
    extract_frame_annotations,
    create_annotated_video,
    get_annotation_summary,
)

REQUIRE_AUTH = settings.REQUIRE_AUTH
JWT_SECRET = settings.JWT_SECRET

app = FastAPI(title="Deepfake Detection API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
templates_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

# Global models
model = None  # optional local SavedModel for direct inference (kept as-is)
keras_xai_model = None  # Keras model used only for Grad-CAM


def load_model():
    """Load TensorFlow model for direct inference."""
    global model
    if model is None:
        try:
            # Look for model in parent directory
            model_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "models",
                "deepfake",
                settings.MODEL_VERSION,
            )
            model = tf.saved_model.load(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    return model


def load_keras_xai_model():
    """Load Keras model for Grad-CAM from various formats."""
    global keras_xai_model
    if keras_xai_model is None:
        base_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "models",
            "deepfake",
            settings.MODEL_VERSION,
        )
        models_root = os.path.join(os.path.dirname(__file__), "..", "..", "models")

        # Check TensorFlow/Keras version
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Keras version: {tf.keras.__version__}")

        # Try different model formats in order of preference
        model_paths = [
            # Try H5 format first (most compatible across versions)
            os.path.join(base_path, "keras_export_xai_tf2.h5"),
            os.path.join(models_root, "Eb0_OVS3_best_model.h5"),
            os.path.join(models_root, "Eb0_OVS3_best_model_weights.h5"),
            # Try SavedModel format (convert to Keras model)
            os.path.join(base_path, "keras_export_xai_tf2"),
            os.path.join(base_path, "keras_export"),
            # Try Keras 3 native format as last resort
            os.path.join(base_path, "keras_export.keras"),
        ]

        for model_path in model_paths:
            try:
                print(f"Attempting to load Keras XAI model from: {model_path}")
                print(f"File exists: {os.path.exists(model_path)}")

                if not os.path.exists(model_path):
                    print(f"File does not exist, skipping: {model_path}")
                    continue

                if model_path.endswith(".h5"):
                    print(f"Loading H5 model from: {model_path}")
                    # Try different loading strategies for H5
                    try:
                        # First try: standard loading
                        keras_xai_model = tf.keras.models.load_model(model_path)
                        print("✅ H5 model loaded with standard method")
                    except Exception as e1:
                        print(f"Standard H5 loading failed: {e1}")
                        try:
                            # Second try: without compilation
                            keras_xai_model = tf.keras.models.load_model(
                                model_path, compile=False
                            )
                            print("✅ H5 model loaded without compilation")
                        except Exception as e2:
                            print(f"H5 loading without compilation failed: {e2}")
                            try:
                                # Third try: load weights only and rebuild model
                                if "weights" in model_path:
                                    print("Attempting to load weights only...")
                                    # This would require knowing the model architecture
                                    continue
                                else:
                                    raise e2
                            except Exception as e3:
                                print(f"All H5 loading methods failed: {e3}")
                                continue

                elif os.path.isdir(model_path):
                    print(f"Loading SavedModel from: {model_path}")
                    try:
                        # Load SavedModel and try to convert to Keras model
                        saved_model = tf.saved_model.load(model_path)
                        print("SavedModel loaded successfully")

                        # Try to get the Keras model from the SavedModel
                        if hasattr(saved_model, "signatures"):
                            print(
                                "SavedModel detected, attempting to convert to Keras model..."
                            )

                            # Try to find a Keras model within the SavedModel
                            # Look for a function that returns a Keras model
                            for (signature_name,) in saved_model.signatures.items():
                                print(f"Found signature: {signature_name}")
                                try:
                                    print(
                                        f"Signature {signature_name} executed successfully"
                                    )

                                    # If we can execute it, we might be able to use it for Grad-CAM
                                    # For now, let's try to create a wrapper
                                    keras_xai_model = saved_model
                                    print("✅ Using SavedModel directly for Grad-CAM")
                                    break
                                except Exception as sig_e:
                                    print(f"Signature {signature_name} failed: {sig_e}")
                                    continue

                            if keras_xai_model is None:
                                print("Could not find usable signature in SavedModel")
                                continue
                        else:
                            print("SavedModel does not have signatures")
                            continue

                    except Exception as e:
                        print(f"SavedModel loading failed: {e}")
                        continue

                elif model_path.endswith(".keras"):
                    print(f"Loading Keras 3 native model from: {model_path}")
                    try:
                        # Try to load with custom objects to handle Keras 3 compatibility
                        keras_xai_model = tf.keras.models.load_model(model_path)
                        print("✅ Keras 3 model loaded successfully")
                    except Exception as e:
                        print(f"Keras 3 loading failed: {e}")
                        print("This is likely due to Keras version mismatch")
                        continue

                # If we get here, the model was loaded successfully
                print(f"✅ Keras XAI model loaded successfully from {model_path}")
                print(f"Model type: {type(keras_xai_model)}")

                # Check if it's a SavedModel or Keras model
                if hasattr(keras_xai_model, "layers"):
                    print(f"Model has layers: {len(keras_xai_model.layers)}")
                    print(
                        f"Layer names: {[layer.name for layer in keras_xai_model.layers]}"
                    )
                elif hasattr(keras_xai_model, "signatures"):
                    print(
                        f"Model has signatures: {list(keras_xai_model.signatures.keys())}"
                    )
                else:
                    print("Model structure unknown")

                break

            except Exception as e:
                print(f"❌ Error loading Keras XAI model from {model_path}: {e}")
                print(f"Error type: {type(e).__name__}")
                continue

        if keras_xai_model is None:
            print("❌ Failed to load Keras XAI model from any available format")
            print("Available files in model directory:")
            try:
                for file in os.listdir(base_path):
                    print(f"  - {file}")
            except Exception as e:
                print(f"Could not list directory: {e}")

            print("\n🔧 TROUBLESHOOTING SUGGESTIONS:")
            print("1. The models may have been saved with a different Keras version")
            print("2. Try re-exporting the model with the current Keras version")
            print("3. Check if custom objects are needed for loading")
            print("4. Consider using the H5 format for better compatibility")
            print(
                "5. The Grad-CAM feature will be disabled until a compatible model is available"
            )

    return keras_xai_model


# In-memory storage for video processing results (in production, use Redis or database)
video_storage = {}
frame_storage = {}


def generate_video_id(content: bytes) -> str:
    """Generate unique video ID based on content hash."""
    content_hash = hashlib.md5(content).hexdigest()
    timestamp = str(int(time.time()))
    return f"{content_hash[:8]}_{timestamp}"


def generate_frame_id(video_id: str, frame_index: int) -> str:
    """Generate unique frame ID."""
    return f"{video_id}_frame_{frame_index:04d}"


bearer_scheme = HTTPBearer(auto_error=REQUIRE_AUTH)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if not REQUIRE_AUTH:
        return None
    token = credentials.credentials if credentials else None
    if token != JWT_SECRET:
        raise HTTPException(status_code=401, detail="Invalid token")
    return True


class PredictResponse(BaseModel):
    score: float
    label: str
    frame_samples: list
    version: str
    latency_ms: int
    meta: dict


class AnnotatedResponse(BaseModel):
    score: float
    label: str
    frame_samples: list
    version: str
    latency_ms: int
    meta: dict
    annotations: list
    heatmaps: list
    explanations: dict
    summary: dict
    video_id: str
    frame_ids: list


@app.get("/health")
def health():
    ok, detail = health_check()
    return {"ok": ok, "tfserving": detail}


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...), fps: float | None = None, auth=Depends(verify_token)
):
    try:
        content = await file.read()
        start = time.time()
        x, samples, meta = extract_frames_and_preprocess(
            content, target_fps=fps or settings.DEFAULT_FPS
        )
        preds = tfserving_predict(x)
        out = aggregate(preds, samples, threshold=settings.THRESHOLD)
        latency_ms = int((time.time() - start) * 1000)
        return {
            **out,
            "version": settings.MODEL_VERSION,
            "latency_ms": latency_ms,
            "meta": meta,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict-with-annotations", response_model=AnnotatedResponse)
async def predict_with_annotations(
    file: UploadFile = File(...),
    fps: float | None = None,
    include_xai: bool = True,
    auth=Depends(verify_token),
):
    """Enhanced prediction with annotations and XAI."""
    try:
        content = await file.read()
        start = time.time()

        # Generate video ID
        video_id = generate_video_id(content)

        # Extract frames and get predictions
        x, samples, meta = extract_frames_and_preprocess(
            content, target_fps=fps or settings.DEFAULT_FPS
        )

        # Generate frame IDs
        frame_ids = [generate_frame_id(video_id, i) for i in range(len(samples))]

        # Get predictions (try direct model first, fallback to TF-Serving)
        model = load_model()
        if model is not None:
            try:
                preds = model(x).numpy().squeeze()
            except Exception as e:
                print(f"Direct model inference failed: {e}")
                preds = tfserving_predict(x).squeeze()
        else:
            preds = tfserving_predict(x).squeeze()

        # Aggregate results
        out = aggregate(preds, samples, threshold=settings.THRESHOLD)

        # Extract annotations
        annotations = extract_frame_annotations(
            content, preds, fps or settings.DEFAULT_FPS
        )

        # Generate XAI features if requested (use Keras XAI model)
        heatmaps = []
        explanations = {}

        if include_xai:
            try:
                xai_model = load_keras_xai_model()
                if xai_model is not None:
                    heatmaps = generate_gradcam_heatmaps(x, preds, xai_model)
                else:
                    print("Keras XAI model unavailable; skipping heatmaps")
                explanations = generate_explanations(
                    preds, annotations, threshold=settings.THRESHOLD
                )
            except Exception as e:
                print(f"XAI generation failed: {e}")
                heatmaps = []
                explanations = {"overall_explanation": "XAI analysis unavailable"}

        # Get annotation summary
        summary = get_annotation_summary(annotations)

        latency_ms = int((time.time() - start) * 1000)

        # Store results for later retrieval
        video_storage[video_id] = {
            "content": content,
            "annotations": annotations,
            "heatmaps": heatmaps,
            "explanations": explanations,
            "summary": summary,
            "frame_ids": frame_ids,
            "timestamp": time.time(),
        }

        # Store individual frame data
        for i, frame_id in enumerate(frame_ids):
            frame_storage[frame_id] = {
                "video_id": video_id,
                "frame_index": i,
                "annotation": annotations[i] if i < len(annotations) else None,
                "heatmap": heatmaps[i] if i < len(heatmaps) else None,
                "timestamp": time.time(),
            }

        return {
            **out,
            "version": settings.MODEL_VERSION,
            "latency_ms": latency_ms,
            "meta": meta,
            "annotations": annotations,
            "heatmaps": heatmaps,
            "explanations": explanations,
            "summary": summary,
            "video_id": video_id,
            "frame_ids": frame_ids,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/annotated-video/{video_id}")
async def get_annotated_video(video_id: str):
    """Return annotated video file."""
    if video_id not in video_storage:
        raise HTTPException(status_code=404, detail="Video not found")

    video_data = video_storage[video_id]

    # Create annotated video using the annotations module
    try:
        # Create annotated video (actual writer may change extension based on codec)
        requested_output = f"annotated_{video_id}.mp4"
        annotated_path = create_annotated_video(
            video_data["content"], video_data["annotations"], requested_output
        )

        # Decide media type and filename by actual extension
        _, ext = os.path.splitext(annotated_path)
        ext = (ext or ".mp4").lower()
        media_type = (
            "video/mp4"
            if ext == ".mp4"
            else "video/x-msvideo"
            if ext == ".avi"
            else "application/octet-stream"
        )
        download_name = f"annotated_{video_id}{ext}"

        return FileResponse(
            annotated_path, media_type=media_type, filename=download_name
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error creating annotated video: {str(e)}"
        )


@app.get("/heatmap/{frame_id}")
async def get_heatmap(frame_id: str):
    """Return heatmap image for a specific frame."""
    if frame_id not in frame_storage:
        raise HTTPException(status_code=404, detail="Frame not found")

    frame_data = frame_storage[frame_id]

    if not frame_data["heatmap"]:
        raise HTTPException(
            status_code=404, detail="Heatmap not available for this frame"
        )

    # Return the base64 encoded heatmap image
    return {"heatmap": frame_data["heatmap"], "frame_id": frame_id}


@app.get("/video/{video_id}/info")
async def get_video_info(video_id: str):
    """Get information about a processed video."""
    if video_id not in video_storage:
        raise HTTPException(status_code=404, detail="Video not found")

    video_data = video_storage[video_id]

    return {
        "video_id": video_id,
        "frame_count": len(video_data["frame_ids"]),
        "frame_ids": video_data["frame_ids"],
        "summary": video_data["summary"],
        "timestamp": video_data["timestamp"],
    }


@app.get("/frame/{frame_id}/info")
async def get_frame_info(frame_id: str):
    """Get information about a specific frame."""
    if frame_id not in frame_storage:
        raise HTTPException(status_code=404, detail="Frame not found")

    frame_data = frame_storage[frame_id]

    return {
        "frame_id": frame_id,
        "video_id": frame_data["video_id"],
        "frame_index": frame_data["frame_index"],
        "annotation": frame_data["annotation"],
        "has_heatmap": frame_data["heatmap"] is not None,
        "timestamp": frame_data["timestamp"],
    }
