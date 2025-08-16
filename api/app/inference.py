import os
import cv2
import json
import tempfile
import numpy as np
import requests
from settings import settings

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


def extract_frames_and_preprocess(
    video_bytes: bytes, target_fps: float = DEFAULT_FPS, max_frames: int = MAX_FRAMES
):
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
        step = max(int(round(src_fps / max(target_fps, 0.1))), 1)

        cascade = _haar_cascade()
        frames, samples = [], []
        idx = out_count = 0
        face_frames = 0

        while True:
            ret = cap.grab()
            if not ret:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break

                ts_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                h, w = frame.shape[:2]

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                if faces is not None and len(faces) > 0:
                    x, y, fw, fh = _largest_face(faces)
                    x, y, fw, fh = _expand_bbox(x, y, fw, fh, w, h, margin=0.20)
                    face = frame[y : y + fh, x : x + fw]

                    patch = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)
                    arr = patch.astype(np.float32) / 255.0

                    frames.append(arr)
                    samples.append((float(ts_sec), out_count))
                    face_frames += 1
                    out_count += 1
                    if out_count >= max_frames:
                        break
            idx += 1

        cap.release()

        if len(frames) == 0:
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 0:
                stride = max(total // min(16, total), 1)
                i = 0
                while i < total and len(frames) < max_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ok, frame = cap.read()
                    if not ok:
                        break
                    patch = cv2.cvtColor(_center_square_crop(frame), cv2.COLOR_BGR2RGB)
                    patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_AREA)
                    arr = patch.astype(np.float32) / 255.0
                    frames.append(arr)
                    samples.append((float(i / (src_fps or 25.0)), len(samples)))
                    i += stride
                cap.release()

        if len(frames) == 0:
            raise ValueError("No frames extracted (no faces and fallback failed)")

        x = np.stack(frames, axis=0)
        meta = {
            "src_fps": src_fps,
            "total_frames": total_frames,
            "used_step": step,
            "face_frames": face_frames,
            "face_detect_rate": float(face_frames) / float(len(frames))
            if len(frames)
            else 0.0,
        }
        return x, samples, meta
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def tfserving_predict(batch: np.ndarray):
    url = f"{TF_SERVING_URL}/v1/models/{MODEL_NAME}:predict"
    payload = {"instances": batch.tolist()}
    resp = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    if resp.status_code != 200:
        raise RuntimeError(f"TF Serving error {resp.status_code}: {resp.text[:400]}")
    data = resp.json()
    preds = data.get("predictions") or data.get("outputs") or data.get("result")
    if preds is None:
        raise RuntimeError(f"Malformed TF Serving response: {json.dumps(data)[:400]}")
    return np.array(preds, dtype=np.float32)


def aggregate(scores: np.ndarray, samples, threshold: float = THRESHOLD):
    s = scores.squeeze()
    video_score = float(np.mean(s))
    label = "fake" if video_score >= threshold else "real"
    frame_samples = [{"t": float(t), "score": float(s[i])} for (t, i) in samples]
    return {"score": video_score, "label": label, "frame_samples": frame_samples}


def health_check():
    try:
        url = f"{TF_SERVING_URL}/v1/models/{MODEL_NAME}"
        r = requests.get(url, timeout=3)
        if r.status_code != 200:
            return False, f"status={r.status_code}"
        return True, r.json()
    except Exception as e:
        return False, str(e)
