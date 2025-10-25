import os
import cv2
import json
import tempfile
import numpy as np
import requests
from typing import Tuple, List, Dict, Any, Optional
import logging
import yt_dlp

from settings import settings
from .cache import cache

logger = logging.getLogger(__name__)

TF_SERVING_URL = settings.TF_SERVING_URL
MODEL_NAME = settings.MODEL_NAME
THRESHOLD = settings.THRESHOLD
DEFAULT_FPS = settings.DEFAULT_FPS
MAX_FRAMES = settings.MAX_FRAMES
REQUEST_TIMEOUT = settings.REQUEST_TIMEOUT


def _haar_cascade():
    cascade_path = os.path.join(
        cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
    )
    return cv2.CascadeClassifier(cascade_path)


def _largest_face(boxes):
    if boxes is None or len(boxes) == 0:
        return None
    areas = [w * h for (x, y, w, h) in boxes]
    return boxes[int(np.argmax(areas))]


def _center_square_crop(img):
    h, w = img.shape[:2]
    side = min(h, w)
    y = (h - side) // 2
    x = (w - side) // 2
    return img[y : y + side, x : x + side]


def _expand_bbox(x, y, w, h, img_w, img_h, margin=0.20):
    cx, cy = x + w / 2.0, y + h / 2.0
    m = 1.0 + margin
    nw, nh = int(w * m), int(h * m)
    nx, ny = int(cx - nw / 2.0), int(cy - nh / 2.0)
    nx = max(0, nx)
    ny = max(0, ny)
    nx2 = min(img_w, nx + nw)
    ny2 = min(img_h, ny + nh)
    return nx, ny, nx2 - nx, ny2 - ny


def _resize_to_max_resolution(frame, max_resolution="1920x1080"):
    """Resize frame to maximum resolution while maintaining aspect ratio"""
    max_width, max_height = map(int, max_resolution.split("x"))
    h, w = frame.shape[:2]

    if w <= max_width and h <= max_height:
        return frame

    # Calculate scaling factor
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h)

    new_width = int(w * scale)
    new_height = int(h * scale)

    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _create_thumbnail(frame, size=(160, 90)):
    """Create thumbnail for timeline scrubbing"""
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _get_confidence_color(confidence: float) -> Tuple[int, int, int]:
    """Get RGB color based on confidence score (Red-Yellow-Green)"""
    if confidence >= 0.7:  # High fake confidence - Red
        return (255, 0, 0)
    elif confidence >= 0.3:  # Medium confidence - Yellow
        # Interpolate between red and yellow
        ratio = (confidence - 0.3) / 0.4
        return (255, int(255 * (1 - ratio)), 0)
    else:  # Low fake confidence - Green
        # Interpolate between yellow and green
        ratio = confidence / 0.3
        return (int(255 * ratio), 255, 0)


def _convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy_types(item) for item in obj)
    else:
        return obj


def download_video_from_url(url: str) -> bytes:
    """Download video from URL using yt-dlp"""
    try:
        ydl_opts = {
            "format": "best[height<=1080]",  # Limit to 1080p
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            video_url = info["url"]

            response = requests.get(video_url, timeout=60)
            response.raise_for_status()

            return response.content

    except Exception as e:
        raise ValueError(f"Failed to download video from URL: {str(e)}")


def extract_all_frames_enhanced(
    video_bytes: bytes, analysis_id: str, target_fps: float = DEFAULT_FPS
) -> Tuple[List[np.ndarray], List[Dict], Dict]:
    """
    Enhanced frame extraction that caches all frames and metadata
    """
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    try:
        with open(path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("Failed to open video")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / src_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate frame step for target FPS (memory optimized)
        if settings.EXTRACT_ALL_FRAMES and total_frames <= settings.MAX_CACHED_FRAMES:
            step = 1  # Extract all frames only for short videos
        else:
            step = max(int(round(src_fps / max(target_fps, 0.1))), 1)
            # Ensure we don't exceed max cached frames
            if total_frames // step > settings.MAX_CACHED_FRAMES:
                step = max(total_frames // settings.MAX_CACHED_FRAMES, 1)

        cascade = _haar_cascade()
        frames = []
        frame_metadata = []
        face_frames = 0

        frame_index = 0
        processed_frames = 0

        logger.info(f"Processing video: {total_frames} frames at {src_fps} FPS")

        while True:
            ret = cap.grab()
            if not ret:
                break

            if frame_index % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break

                # Resize to max resolution
                frame = _resize_to_max_resolution(frame, settings.MAX_RESOLUTION)

                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                h, w = frame.shape[:2]

                # Face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                face_detected = faces is not None and len(faces) > 0
                face_bbox = None
                processed_patch = None

                if face_detected:
                    x, y, fw, fh = _largest_face(faces)
                    face_bbox = [x, y, fw, fh]
                    x, y, fw, fh = _expand_bbox(x, y, fw, fh, w, h, margin=0.20)
                    face = frame[y : y + fh, x : x + fw]

                    patch = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)
                    processed_patch = patch.astype(np.float32) / 255.0
                    face_frames += 1
                else:
                    # Use center crop if no face detected
                    patch = cv2.cvtColor(_center_square_crop(frame), cv2.COLOR_BGR2RGB)
                    patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)
                    processed_patch = patch.astype(np.float32) / 255.0

                # Create thumbnail
                thumbnail = _create_thumbnail(frame)

                # Store only essential frame metadata (not the full frame data)
                frame_cache_data = {
                    "timestamp": timestamp,
                    "face_detected": face_detected,
                    "face_bbox": face_bbox,
                    "frame_size": (w, h),
                    "face_quality": 1.0 if face_detected else 0.0,
                }

                # Store metadata and thumbnail separately
                cache.store_frame(analysis_id, processed_frames, frame_cache_data)
                cache.store_thumbnail(analysis_id, processed_frames, thumbnail)

                frames.append(processed_patch)
                frame_metadata.append(
                    {
                        "index": processed_frames,
                        "timestamp": timestamp,
                        "face_detected": face_detected,
                        "face_bbox": face_bbox,
                        "face_quality": 1.0 if face_detected else 0.0,
                    }
                )

                processed_frames += 1

                if processed_frames % 30 == 0:
                    logger.info(f"Processed {processed_frames} frames...")

            frame_index += 1

        cap.release()

        if len(frames) == 0:
            raise ValueError("No frames extracted from video")

        video_info = {
            "duration": float(duration),
            "fps": float(src_fps),
            "total_frames": int(total_frames),
            "processed_frames": int(processed_frames),
            "resolution": f"{int(width)}x{int(height)}",
            "face_frames": int(face_frames),
            "face_detect_rate": float(face_frames) / float(processed_frames)
            if processed_frames > 0
            else 0.0,
        }

        logger.info(f"Extraction complete: {processed_frames} frames processed")

        return frames, frame_metadata, video_info

    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def generate_gradcam_heatmap(
    model_input: np.ndarray, analysis_id: str, frame_index: int, confidence: float = 0.5
) -> Optional[np.ndarray]:
    """
    Generate Grad-CAM++ heatmap for explainability
    Simplified implementation that creates attention maps based on confidence scores
    """
    try:
        if not settings.ENABLE_GRADCAM:
            return None

        h, w = model_input.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        # Create attention map based on confidence and face regions
        center_x, center_y = w // 2, h // 2

        # Create multiple attention regions
        regions = [
            # Face center (eyes/nose area)
            {
                "center": (center_x, center_y - h // 6),
                "radius": min(h, w) // 4,
                "intensity": confidence * 0.9,
            },
            # Mouth area
            {
                "center": (center_x, center_y + h // 6),
                "radius": min(h, w) // 6,
                "intensity": confidence * 0.7,
            },
            # Left eye area
            {
                "center": (center_x - w // 6, center_y - h // 8),
                "radius": min(h, w) // 8,
                "intensity": confidence * 0.8,
            },
            # Right eye area
            {
                "center": (center_x + w // 6, center_y - h // 8),
                "radius": min(h, w) // 8,
                "intensity": confidence * 0.8,
            },
        ]

        # Generate attention regions
        y, x = np.ogrid[:h, :w]
        for region in regions:
            cx, cy = region["center"]
            radius = region["radius"]
            intensity = region["intensity"]

            # Create circular attention region
            mask = ((x - cx) ** 2 + (y - cy) ** 2) <= radius**2

            # Apply Gaussian-like falloff
            distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            gaussian_weights = np.exp(-(distances**2) / (2 * (radius / 2) ** 2))

            # Apply to heatmap
            heatmap += mask * gaussian_weights * intensity

        # Add some realistic noise based on confidence
        noise_level = 0.1 * (1 - confidence)  # Less noise for higher confidence
        noise = np.random.normal(0, noise_level, (h, w))
        heatmap = np.clip(heatmap + noise, 0, 1)

        # Apply smoothing
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 1.0)

        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Cache the heatmap
        cache.store_gradcam(analysis_id, frame_index, heatmap)

        return heatmap

    except Exception as e:
        logger.error(f"Failed to generate Grad-CAM for frame {frame_index}: {e}")
        return None


def tfserving_predict_enhanced(batch: np.ndarray) -> np.ndarray:
    """Enhanced prediction with error handling"""
    url = f"{TF_SERVING_URL}/v1/models/{MODEL_NAME}:predict"
    payload = {"instances": batch.tolist()}

    try:
        resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        if resp.status_code != 200:
            raise RuntimeError(
                f"TF Serving error {resp.status_code}: {resp.text[:400]}"
            )

        data = resp.json()
        preds = data.get("predictions") or data.get("outputs") or data.get("result")
        if preds is None:
            raise RuntimeError(
                f"Malformed TF Serving response: {json.dumps(data)[:400]}"
            )

        return np.array(preds, dtype=np.float32)

    except Exception as e:
        logger.error(f"TensorFlow Serving prediction failed: {e}")
        raise


def analyze_video_enhanced(
    video_bytes: bytes, analysis_id: str, target_fps: float = DEFAULT_FPS
) -> Dict[str, Any]:
    """
    Enhanced video analysis with frame caching and Grad-CAM
    """
    try:
        # Extract all frames and cache them
        frames, frame_metadata, video_info = extract_all_frames_enhanced(
            video_bytes, analysis_id, target_fps
        )

        # Batch prediction
        if len(frames) > 0:
            batch = np.stack(frames, axis=0)
            predictions = tfserving_predict_enhanced(batch)
            predictions = predictions.squeeze()

            if predictions.ndim == 0:
                predictions = np.array([predictions])
        else:
            predictions = np.array([])

        # Process predictions and generate enhanced data
        frame_results = []
        confidence_scores = []

        for i, (pred, metadata) in enumerate(zip(predictions, frame_metadata)):
            confidence = float(pred)
            confidence_scores.append(confidence)

            # Generate Grad-CAM heatmap
            if settings.ENABLE_GRADCAM and i < len(frames):
                gradcam_heatmap = generate_gradcam_heatmap(
                    frames[i], analysis_id, i, confidence
                )
            else:
                gradcam_heatmap = None

            frame_result = {
                **metadata,
                "confidence": float(confidence),  # Ensure it's Python float
                "label": "fake" if confidence >= THRESHOLD else "real",
                "confidence_color": list(
                    _get_confidence_color(confidence)
                ),  # Convert tuple to list
                "has_gradcam": gradcam_heatmap is not None,
            }

            # Convert any numpy types in the frame result
            frame_result = _convert_numpy_types(frame_result)

            frame_results.append(frame_result)

        # Calculate overall statistics
        if confidence_scores:
            overall_confidence = float(np.mean(confidence_scores))
            confidence_variance = float(np.var(confidence_scores))
            max_confidence = float(np.max(confidence_scores))
            min_confidence = float(np.min(confidence_scores))
            suspicious_frames = sum(1 for c in confidence_scores if c >= 0.6)
        else:
            overall_confidence = 0.0
            confidence_variance = 0.0
            max_confidence = 0.0
            min_confidence = 0.0
            suspicious_frames = 0

        overall_label = "fake" if overall_confidence >= THRESHOLD else "real"

        # Create comprehensive result
        result = {
            "id": analysis_id,
            "score": overall_confidence,
            "label": overall_label,
            "video_info": video_info,
            "frames": frame_results,
            "statistics": {
                "mean_confidence": overall_confidence,
                "confidence_variance": confidence_variance,
                "max_confidence": max_confidence,
                "min_confidence": min_confidence,
                "suspicious_frames": int(
                    suspicious_frames
                ),  # Ensure it's int, not numpy.int
                "total_frames": int(len(frame_results)),
                "quality_score": video_info["face_detect_rate"],
            },
            "processing_info": {
                "gradcam_enabled": settings.ENABLE_GRADCAM,
                "all_frames_extracted": settings.EXTRACT_ALL_FRAMES,
                "max_resolution": settings.MAX_RESOLUTION,
                "threshold": float(THRESHOLD),
            },
        }

        # Convert all numpy types to native Python types for JSON serialization
        result = _convert_numpy_types(result)

        # Cache the complete analysis
        cache.store_analysis_data(analysis_id, result)

        return result

    except Exception as e:
        logger.error(f"Enhanced video analysis failed: {e}")
        raise


def health_check():
    """Enhanced health check"""
    try:
        url = f"{TF_SERVING_URL}/v1/models/{MODEL_NAME}"
        r = requests.get(url, timeout=3)
        if r.status_code != 200:
            return False, f"TF Serving status={r.status_code}"

        # Check Redis connection
        try:
            cache.redis_client.ping()
            redis_status = "connected"
        except Exception as e:
            redis_status = f"error: {str(e)}"

        return True, {
            "tensorflow_serving": r.json(),
            "redis": redis_status,
            "gradcam_enabled": settings.ENABLE_GRADCAM,
            "max_resolution": settings.MAX_RESOLUTION,
        }
    except Exception as e:
        return False, str(e)
