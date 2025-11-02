import redis  # type: ignore
import json
import base64
import numpy as np
import cv2
from typing import Optional, List, Dict, Any
from settings import settings
import logging

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self):
        self.redis_client = redis.from_url(
            settings.REDIS_URL,
            db=settings.REDIS_DB,
            decode_responses=False,  # We'll handle binary data
        )

    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode numpy array frame to base64 string"""
        return base64.b64encode(frame.tobytes()).decode("utf-8")

    def _decode_frame(self, encoded_frame: str, shape: tuple, dtype: str) -> np.ndarray:
        """Decode base64 string back to numpy array"""
        frame_bytes = base64.b64decode(encoded_frame.encode("utf-8"))
        return np.frombuffer(frame_bytes, dtype=dtype).reshape(shape)

    def store_analysis_data(self, analysis_id: str, data: Dict[str, Any]) -> bool:
        """Store complete analysis data"""
        try:
            key = f"analysis:{analysis_id}"
            serialized_data = json.dumps(data, default=str)
            self.redis_client.setex(key, settings.FRAME_CACHE_TTL, serialized_data)
            return True
        except Exception as e:
            logger.error(f"Failed to store analysis data: {e}")
            return False

    def get_analysis_data(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve complete analysis data"""
        try:
            key = f"analysis:{analysis_id}"
            data = self.redis_client.get(key)
            if data:
                return json.loads(data.decode("utf-8"))  # type: ignore
            return None
        except Exception as e:
            logger.error(f"Failed to get analysis data: {e}")
            return None

    def store_frame(
        self, analysis_id: str, frame_index: int, frame_data: Dict[str, Any]
    ) -> bool:
        """Store individual frame data including image and metadata (memory optimized)"""
        try:
            key = f"frame:{analysis_id}:{frame_index}"

            # Only store essential metadata, not the full frame data
            essential_data = {
                "timestamp": frame_data.get("timestamp", 0.0),
                "face_detected": frame_data.get("face_detected", False),
                "face_bbox": frame_data.get("face_bbox"),
                "frame_size": frame_data.get("frame_size"),
                "face_quality": frame_data.get("face_quality", 0.0),
            }

            # Skip storing large numpy arrays to save memory
            # Only store thumbnails and essential metadata

            serialized_data = json.dumps(essential_data, default=str)
            self.redis_client.setex(key, settings.FRAME_CACHE_TTL, serialized_data)
            return True
        except Exception as e:
            logger.error(f"Failed to store frame {frame_index}: {e}")
            return False

    def get_frame(self, analysis_id: str, frame_index: int) -> Optional[Dict[str, Any]]:
        """Retrieve individual frame data"""
        try:
            key = f"frame:{analysis_id}:{frame_index}"
            data = self.redis_client.get(key)
            if data:
                frame_data = json.loads(data.decode("utf-8"))

                # Reconstruct numpy arrays
                processed_data = {}
                for k, v in frame_data.items():
                    if isinstance(v, dict) and v.get("is_array"):
                        processed_data[k] = self._decode_frame(
                            v["data"], tuple(v["shape"]), v["dtype"]
                        )
                    else:
                        processed_data[k] = v

                return processed_data
            return None
        except Exception as e:
            logger.error(f"Failed to get frame {frame_index}: {e}")
            return None

    def store_gradcam(
        self, analysis_id: str, frame_index: int, gradcam_data: np.ndarray
    ) -> bool:
        """Store Grad-CAM heatmap for a specific frame"""
        try:
            key = f"gradcam:{analysis_id}:{frame_index}"
            encoded_data = self._encode_frame(gradcam_data)

            gradcam_info = {
                "data": encoded_data,
                "shape": gradcam_data.shape,
                "dtype": str(gradcam_data.dtype),
            }

            serialized_data = json.dumps(gradcam_info)
            self.redis_client.setex(key, settings.FRAME_CACHE_TTL, serialized_data)
            return True
        except Exception as e:
            logger.error(f"Failed to store Grad-CAM for frame {frame_index}: {e}")
            return False

    def get_gradcam(self, analysis_id: str, frame_index: int) -> Optional[np.ndarray]:
        """Retrieve Grad-CAM heatmap for a specific frame"""
        try:
            key = f"gradcam:{analysis_id}:{frame_index}"
            data = self.redis_client.get(key)
            if data:
                gradcam_info = json.loads(data.decode("utf-8"))
                return self._decode_frame(
                    gradcam_info["data"],
                    tuple(gradcam_info["shape"]),
                    gradcam_info["dtype"],
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get Grad-CAM for frame {frame_index}: {e}")
            return None

    def store_thumbnail(
        self, analysis_id: str, frame_index: int, thumbnail: np.ndarray
    ) -> bool:
        """Store thumbnail for timeline scrubbing (with memory limit handling)"""
        try:
            # Skip storing thumbnails if we're running low on memory
            if frame_index > getattr(settings, "MAX_CACHED_FRAMES", 100):
                return False

            # Check Redis memory usage before storing
            try:
                info = self.redis_client.info("memory")
                used_memory = info.get("used_memory", 0)
                maxmemory = info.get("maxmemory", 0)
                if maxmemory > 0 and used_memory > maxmemory * 0.95:
                    logger.warning("Redis memory > 95% full, skipping thumbnail")
                    return False
            except Exception as e:
                logger.warning(f"Could not check Redis memory: {e}")

            key = f"thumb:{analysis_id}:{frame_index}"

            # Reduce thumbnail size to save memory - balance quality vs memory
            small_thumbnail = cv2.resize(
                thumbnail, (320, 180), interpolation=cv2.INTER_AREA
            )
            encoded_data = self._encode_frame(small_thumbnail)

            thumb_info = {
                "data": encoded_data,
                "shape": small_thumbnail.shape,
                "dtype": str(small_thumbnail.dtype),
            }

            serialized_data = json.dumps(thumb_info)
            self.redis_client.setex(key, settings.FRAME_CACHE_TTL, serialized_data)
            return True
        except Exception as e:
            # Don't log memory errors as errors - they're expected
            if "maxmemory" in str(e).lower():
                logger.warning(
                    f"Redis memory limit reached, skipping thumbnail {frame_index}"
                )
            else:
                logger.error(f"Failed to store thumbnail for frame {frame_index}: {e}")
            return False

    def get_thumbnail(self, analysis_id: str, frame_index: int) -> Optional[np.ndarray]:
        """Retrieve thumbnail for a specific frame"""
        try:
            key = f"thumb:{analysis_id}:{frame_index}"
            data = self.redis_client.get(key)
            if data:
                thumb_info = json.loads(data.decode("utf-8"))
                return self._decode_frame(
                    thumb_info["data"], tuple(thumb_info["shape"]), thumb_info["dtype"]
                )
            return None
        except Exception as e:
            logger.error(f"Failed to get thumbnail for frame {frame_index}: {e}")
            return None

    def cleanup_analysis(self, analysis_id: str) -> bool:
        """Clean up all data related to an analysis"""
        try:
            # Get all keys related to this analysis
            patterns = [
                f"analysis:{analysis_id}",
                f"frame:{analysis_id}:*",
                f"gradcam:{analysis_id}:*",
                f"thumb:{analysis_id}:*",
            ]

            deleted_count = 0
            for pattern in patterns:
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted_count += self.redis_client.delete(*keys)

            logger.info(f"Cleaned up {deleted_count} keys for analysis {analysis_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup analysis {analysis_id}: {e}")
            return False

    def get_analysis_list(self) -> List[str]:
        """Get list of all analysis IDs"""
        try:
            keys = self.redis_client.keys("analysis:*")
            return [key.decode("utf-8").split(":")[1] for key in keys]
        except Exception as e:
            logger.error(f"Failed to get analysis list: {e}")
            return []


# Global cache instance
cache = RedisCache()
