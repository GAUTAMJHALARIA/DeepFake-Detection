from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Core model serving
    MODEL_NAME: str = "deepfake"
    MODEL_VERSION: str = "1"
    TF_SERVING_URL: str = "http://tfserving:8501"

    # Inference controls
    DEFAULT_FPS: float = 2.0
    MAX_FRAMES: int = 256
    THRESHOLD: float = 0.5
    REQUEST_TIMEOUT: float = 30.0

    # Auth
    REQUIRE_AUTH: bool = False
    JWT_SECRET: str = "change-me"

    class Config:
        env_file = ".env"


settings = Settings()
