from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Core model serving
    MODEL_NAME: str = "deepfake"
    MODEL_VERSION: str = "1"
    TF_SERVING_URL: str = "http://localhost:8501"

    # Inference controls
    DEFAULT_FPS: float = 2.0
    MAX_FRAMES: int = 256
    THRESHOLD: float = 0.5
    REQUEST_TIMEOUT: float = 30.0

    # Enhanced features
    MAX_RESOLUTION: str = "1920x1080"  # 1080p limit
    EXTRACT_ALL_FRAMES: bool = False  # Set to False to save memory
    ENABLE_GRADCAM: bool = True
    FRAME_CACHE_TTL: int = 1800  # 30 minutes in seconds (reduced)
    MAX_CACHED_FRAMES: int = 100  # Limit frames to cache

    # Redis configuration
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_DB: int = 0

    # File storage
    TEMP_STORAGE_PATH: str = "/tmp/deepfake_analysis"
    CLEANUP_AFTER_ANALYSIS: bool = True

    # Auth
    REQUIRE_AUTH: bool = False
    JWT_SECRET: str = "change-me"

    class Config:
        env_file = ".env"


settings = Settings()
