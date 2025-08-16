import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from settings import settings
from .inference import (
    extract_frames_and_preprocess,
    tfserving_predict,
    aggregate,
    health_check,
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
