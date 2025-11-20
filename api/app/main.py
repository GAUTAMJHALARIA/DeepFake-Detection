import time
import uuid
import numpy as np
import base64
import cv2
import os
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import logging

from settings import settings
from .inference import (
    get_supported_formats,
)
from .enhanced_inference import (
    analyze_video_enhanced,
    download_video_from_url,
    health_check as enhanced_health_check,
)
from .cache import cache

logger = logging.getLogger(__name__)

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


bearer_scheme = HTTPBearer(auto_error=REQUIRE_AUTH)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    if not REQUIRE_AUTH:
        return None
    token = credentials.credentials if credentials else None
    if token != JWT_SECRET:
        raise HTTPException(status_code=401, detail="Invalid token")
    return True


class EnhancedPredictResponse(BaseModel):
    id: str
    score: float
    label: str
    video_info: dict
    frames: list
    statistics: dict
    processing_info: dict
    latency_ms: int


class FrameResponse(BaseModel):
    frame_index: int
    timestamp: float
    confidence: float
    label: str
    face_detected: bool
    thumbnail_base64: Optional[str] = None


class GradCAMResponse(BaseModel):
    frame_index: int
    heatmap_base64: str
    overlay_base64: Optional[str] = None


@app.post("/predict", response_model=EnhancedPredictResponse)
async def predict(
    file: UploadFile = File(...), fps: float | None = None, auth=Depends(verify_token)
):
    """Advanced video analysis with frame-by-frame detection, caching, and Grad-CAM"""
    analysis_id = str(uuid.uuid4())
    try:
        content = await file.read()
        start = time.time()

        logger.info(f"Starting enhanced analysis for {file.filename}")

        result = analyze_video_enhanced(
            content, analysis_id, target_fps=fps or settings.DEFAULT_FPS
        )

        latency_ms = int((time.time() - start) * 1000)
        result["latency_ms"] = latency_ms
        result["version"] = settings.MODEL_VERSION

        logger.info(f"Enhanced analysis complete in {latency_ms}ms")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced analysis failed: {e}")
        # Cleanup on failure
        cache.cleanup_analysis(analysis_id)
        raise HTTPException(status_code=400, detail=str(e))


class URLRequest(BaseModel):
    url: str
    fps: float | None = None


@app.post("/predict-url")
async def predict_url(request: URLRequest, auth=Depends(verify_token)):
    """Advanced analysis from URL with full feature set"""
    analysis_id = str(uuid.uuid4())
    try:
        logger.info(f"Downloading video from URL: {request.url}")
        video_content = download_video_from_url(request.url)

        start = time.time()
        result = analyze_video_enhanced(
            video_content, analysis_id, target_fps=request.fps or settings.DEFAULT_FPS
        )

        latency_ms = int((time.time() - start) * 1000)
        result["latency_ms"] = latency_ms
        result["version"] = settings.MODEL_VERSION
        result["source_url"] = str(request.url)

        return result

    except Exception as e:
        logger.error(f"URL analysis failed: {e}")
        cache.cleanup_analysis(analysis_id)
        raise HTTPException(status_code=400, detail=f"Failed to process URL: {str(e)}")


@app.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str, auth=Depends(verify_token)):
    """Get cached analysis results"""
    result = cache.get_analysis_data(analysis_id)
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return result


@app.get("/frames/{analysis_id}/{frame_index}", response_model=FrameResponse)
async def get_frame(analysis_id: str, frame_index: int, auth=Depends(verify_token)):
    """Get specific frame data with thumbnail"""
    frame_data = cache.get_frame(analysis_id, frame_index)
    if not frame_data:
        raise HTTPException(status_code=404, detail="Frame not found")

    # Get thumbnail
    thumbnail = cache.get_thumbnail(analysis_id, frame_index)
    thumbnail_base64 = None
    if thumbnail is not None:
        # Convert thumbnail to base64
        _, buffer = cv2.imencode(".png", thumbnail)
        thumbnail_base64 = base64.b64encode(buffer).decode("utf-8")

    return FrameResponse(
        frame_index=frame_index,
        timestamp=frame_data.get("timestamp", 0.0),
        confidence=frame_data.get("confidence", 0.0),
        label=frame_data.get("label", "unknown"),
        face_detected=frame_data.get("face_detected", False),
        thumbnail_base64=thumbnail_base64,
    )


@app.get("/gradcam/{analysis_id}/{frame_index}", response_model=GradCAMResponse)
async def get_gradcam(analysis_id: str, frame_index: int, auth=Depends(verify_token)):
    """Get Grad-CAM heatmap for specific frame"""
    heatmap = cache.get_gradcam(analysis_id, frame_index)
    if heatmap is None:
        raise HTTPException(status_code=404, detail="Grad-CAM data not found")

    # Convert heatmap to base64
    heatmap_normalized = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    _, buffer = cv2.imencode(".png", heatmap_colored)
    heatmap_base64 = base64.b64encode(buffer).decode("utf-8")

    return GradCAMResponse(frame_index=frame_index, heatmap_base64=heatmap_base64)


@app.get("/thumbnails/{analysis_id}")
async def get_all_thumbnails(analysis_id: str, auth=Depends(verify_token)):
    """Get all thumbnails for timeline scrubbing"""
    analysis_data = cache.get_analysis_data(analysis_id)
    if not analysis_data:
        raise HTTPException(status_code=404, detail="Analysis not found")

    thumbnails = []
    total_frames = len(analysis_data.get("frames", []))

    for i in range(total_frames):
        thumbnail = cache.get_thumbnail(analysis_id, i)
        if thumbnail is not None:
            _, buffer = cv2.imencode(".png", thumbnail)
            thumbnail_base64 = base64.b64encode(buffer).decode("utf-8")

            frame_info = analysis_data["frames"][i]
            thumbnails.append(
                {
                    "frame_index": i,
                    "timestamp": frame_info.get("timestamp", 0.0),
                    "confidence": frame_info.get("confidence", 0.0),
                    "thumbnail_base64": thumbnail_base64,
                }
            )

    return {"thumbnails": thumbnails}


@app.delete("/analysis/{analysis_id}")
async def cleanup_analysis(analysis_id: str, auth=Depends(verify_token)):
    """Clean up analysis data from cache"""
    success = cache.cleanup_analysis(analysis_id)
    if success:
        return {"message": f"Analysis {analysis_id} cleaned up successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to cleanup analysis")


@app.get("/video/{analysis_id}")
async def get_converted_video(analysis_id: str):
    """Get the converted video file for preview (no auth required)"""
    try:
        logger.info(f"Video request for analysis_id: {analysis_id}")

        # Get analysis data to check if video was converted
        analysis_data = cache.get_analysis_data(analysis_id)
        if not analysis_data:
            logger.error(f"Analysis not found: {analysis_id}")
            raise HTTPException(status_code=404, detail="Analysis not found")

        conversion_info = analysis_data.get("video_info", {}).get("conversion_info", {})
        final_path = conversion_info.get("final_path")

        logger.info(f"Conversion info: {conversion_info}")
        logger.info(f"Final path: {final_path}")

        if not final_path:
            logger.error("No final_path in conversion_info")
            raise HTTPException(status_code=404, detail="No converted video path found")

        if not os.path.exists(final_path):
            logger.error(f"Converted video file does not exist: {final_path}")
            raise HTTPException(
                status_code=404, detail="Converted video file not found"
            )

        logger.info(f"Serving converted video: {final_path}")

        # Return the converted video file
        from fastapi.responses import FileResponse

        return FileResponse(
            final_path, media_type="video/mp4", filename=f"converted_{analysis_id}.mp4"
        )

    except Exception as e:
        logger.error(f"Error serving converted video: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving video: {str(e)}")


@app.get("/health")
def health():
    """Enhanced health check"""
    ok, detail = enhanced_health_check()
    return {"ok": ok, "details": detail}


@app.get("/supported-formats")
async def supported_formats():
    """Get list of supported video/image formats"""
    return get_supported_formats()
