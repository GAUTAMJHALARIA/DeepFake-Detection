"""
Video annotation module for deepfake detection.
Handles face detection, bounding boxes, and video processing.
"""

import cv2
import numpy as np
import tempfile
import os
from typing import List, Dict, Tuple, Any
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from settings import settings


def extract_frame_annotations(
    video_bytes: bytes,
    predictions: np.ndarray,
    target_fps: float = 2.0,
    max_frames: int = 256,
) -> List[Dict]:
    """
    Extract annotation data for each frame with face detection.

    Args:
        video_bytes: Video file bytes
        predictions: Model predictions for frames
        target_fps: Target FPS for frame extraction
        max_frames: Maximum number of frames to process

    Returns:
        List of annotation dictionaries
    """
    print(
        f"Extracting annotations: {len(predictions)} predictions, target_fps={target_fps}"
    )
    annotations: List[Dict[str, Any]] = []

    # Create temporary video file
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    try:
        with open(path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError("Failed to open video")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(int(round(src_fps / max(target_fps, 0.1))), 1)

        # Load face cascade
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        frame_idx = 0
        annotation_idx = 0

        while True:
            ret = cap.grab()
            if not ret:
                break

            if frame_idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break

                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                h, w = frame.shape[:2]

                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                # Get prediction for this frame
                prediction = (
                    predictions[annotation_idx]
                    if annotation_idx < len(predictions)
                    else 0.0
                )

                # Create annotation
                annotation: Dict[str, Any] = {
                    "frame_index": frame_idx,
                    "timestamp": float(timestamp),
                    "prediction": float(prediction),
                    "faces": [],  # List[Dict[str, Any]]
                    "frame_dimensions": {"width": w, "height": h},
                }

                # Process detected faces
                if faces is not None and len(faces) > 0:
                    for face_idx, (x, y, fw, fh) in enumerate(faces):
                        # Expand bounding box
                        expanded_bbox = _expand_bbox(x, y, fw, fh, w, h, margin=0.20)

                        face_annotation = {
                            "face_id": face_idx,
                            "bbox": {
                                "x": int(expanded_bbox[0]),
                                "y": int(expanded_bbox[1]),
                                "width": int(expanded_bbox[2]),
                                "height": int(expanded_bbox[3]),
                            },
                            "confidence": float(prediction),
                            "label": "fake"
                            if prediction >= settings.THRESHOLD
                            else "real",
                        }

                        if "faces" not in annotation:
                            annotation["faces"] = []
                        annotation["faces"].append(face_annotation)

                annotations.append(annotation)
                annotation_idx += 1

                if annotation_idx >= max_frames:
                    break

            frame_idx += 1

        cap.release()

    finally:
        try:
            os.remove(path)
        except Exception:
            pass

    print(f"Extracted {len(annotations)} annotations")
    return annotations


def create_annotated_frame(
    frame: np.ndarray,
    annotation: Dict,
    show_confidence: bool = True,
    show_labels: bool = True,
) -> np.ndarray:
    """
    Create annotated frame with bounding boxes and labels.

    Args:
        frame: Original frame
        annotation: Annotation data for the frame
        show_confidence: Whether to show confidence scores
        show_labels: Whether to show labels

    Returns:
        Annotated frame
    """
    annotated_frame = frame.copy()

    # Get annotation colors
    colors = {
        "fake": (0, 0, 255),  # Red
        "real": (0, 255, 0),  # Green
        "uncertain": (0, 255, 255),  # Yellow
    }

    for face in annotation.get("faces", []):
        bbox = face["bbox"]
        confidence = face["confidence"]
        label = face["label"]

        # Determine color based on confidence
        if confidence >= settings.THRESHOLD + 0.1:
            color = colors["fake"]
        elif confidence <= settings.THRESHOLD - 0.1:
            color = colors["real"]
        else:
            color = colors["uncertain"]

        # Draw bounding box
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)

        # Draw label and confidence
        if show_labels or show_confidence:
            label_text = f"{label.upper()}" if show_labels else ""
            if show_confidence:
                conf_text = f" {confidence:.2f}"
                label_text += conf_text

            # Calculate text position
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = x
            text_y = y - 10 if y - 10 > 10 else y + h + 20

            # Draw text background
            cv2.rectangle(
                annotated_frame,
                (text_x, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 10, text_y + 5),
                color,
                -1,
            )

            # Draw text
            cv2.putText(
                annotated_frame,
                label_text,
                (text_x + 5, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

    return annotated_frame


def create_annotated_video(
    video_bytes: bytes, annotations: List[Dict], output_path: str, fps: float = 2.0
) -> str:
    """
    Create annotated video with bounding boxes and labels.

    Args:
        video_bytes: Original video bytes
        annotations: List of frame annotations
        output_path: Output video path
        fps: Output video FPS

    Returns:
        Path to created annotated video
    """
    print(f"Creating annotated video with {len(annotations)} annotations")
    print(f"Output path: {output_path}")

    # Create temporary input file
    fd, input_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)

    try:
        with open(input_path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Failed to open input video")

        # Get video properties
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(
            f"Video properties: {width}x{height}, {src_fps} FPS, {total_frames} frames"
        )

        # Setup video writer - try different codecs and formats
        codec_options = [
            ("mp4v", ".mp4"),
            ("XVID", ".avi"),
            ("MJPG", ".avi"),
            ("H264", ".mp4"),
        ]

        out = None
        final_output_path = output_path

        for codec, ext in codec_options:
            # Try with different extension if needed
            if not output_path.endswith(ext):
                test_path = output_path.rsplit(".", 1)[0] + ext
            else:
                test_path = output_path

            out = cv2.VideoWriter(
                test_path, cv2.VideoWriter_fourcc(*codec), fps, (width, height)
            )
            if out.isOpened():
                print(
                    f"Video writer opened successfully with codec: {codec}, extension: {ext}"
                )
                final_output_path = test_path
                break
            else:
                print(f"Failed to open video writer with codec: {codec}")
                out.release()
                out = None

        if out is None or not out.isOpened():
            # Fallback: try with default settings
            print("Trying fallback video writer...")
            out = cv2.VideoWriter(output_path, -1, fps, (width, height))
            if not out.isOpened():
                raise ValueError(
                    f"Failed to create video writer for {output_path} with any codec"
                )
            else:
                print("Fallback video writer opened successfully")
                final_output_path = output_path

        # Create annotation lookup
        annotation_lookup = {ann["frame_index"]: ann for ann in annotations}
        print(f"Annotation lookup created with {len(annotation_lookup)} entries")

        frame_idx = 0
        frames_written = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Check if we have annotation for this frame
            if frame_idx in annotation_lookup:
                annotation = annotation_lookup[frame_idx]
                annotated_frame = create_annotated_frame(frame, annotation)
                out.write(annotated_frame)
                frames_written += 1
            else:
                out.write(frame)
                frames_written += 1

            frame_idx += 1

        cap.release()
        out.release()

        print(f"Annotated video created: {frames_written} frames written")

        # Check if file was created and has content
        if os.path.exists(final_output_path):
            file_size = os.path.getsize(final_output_path)
            print(f"Output file size: {file_size} bytes")
            if file_size == 0:
                raise ValueError("Output video file is empty")
        else:
            raise ValueError("Output video file was not created")

        return final_output_path

    finally:
        try:
            os.remove(input_path)
        except Exception:
            pass


def draw_confidence_timeline(
    annotations: List[Dict], width: int = 800, height: int = 200
) -> np.ndarray:
    """
    Create a confidence timeline visualization.

    Args:
        annotations: List of frame annotations
        width: Timeline width
        height: Timeline height

    Returns:
        Timeline image as numpy array
    """
    if not annotations:
        return np.zeros((height, width, 3), dtype=np.uint8)

    # Extract timestamps and predictions
    timestamps = [ann["timestamp"] for ann in annotations]
    predictions = [ann["prediction"] for ann in annotations]

    # Normalize timestamps to 0-1
    if len(timestamps) > 1:
        min_time = min(timestamps)
        max_time = max(timestamps)
        if max_time > min_time:
            normalized_times = [
                (t - min_time) / (max_time - min_time) for t in timestamps
            ]
        else:
            normalized_times = [0.5] * len(timestamps)
    else:
        normalized_times = [0.5]

    # Create timeline image
    timeline = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw background
    cv2.rectangle(timeline, (0, 0), (width, height), (50, 50, 50), -1)

    # Draw threshold line
    threshold_y = int(height * (1 - settings.THRESHOLD))
    cv2.line(timeline, (0, threshold_y), (width, threshold_y), (255, 255, 255), 1)

    # Draw confidence curve
    if len(normalized_times) > 1:
        for i in range(len(normalized_times) - 1):
            x1 = int(normalized_times[i] * width)
            x2 = int(normalized_times[i + 1] * width)
            y1 = int(height * (1 - predictions[i]))
            y2 = int(height * (1 - predictions[i + 1]))

            # Color based on prediction
            color = (0, 0, 255) if predictions[i] >= settings.THRESHOLD else (0, 255, 0)
            cv2.line(timeline, (x1, y1), (x2, y2), color, 2)

    # Draw points
    for i, (norm_time, pred) in enumerate(zip(normalized_times, predictions)):
        x = int(norm_time * width)
        y = int(height * (1 - pred))
        color = (0, 0, 255) if pred >= settings.THRESHOLD else (0, 255, 0)
        cv2.circle(timeline, (x, y), 3, color, -1)

    return timeline


def _expand_bbox(
    x: int, y: int, w: int, h: int, img_w: int, img_h: int, margin: float = 0.20
) -> Tuple[int, int, int, int]:
    """Expand bounding box with margin."""
    cx, cy = x + w / 2.0, y + h / 2.0
    m = 1.0 + margin
    nw, nh = int(w * m), int(h * m)
    nx, ny = int(cx - nw / 2.0), int(cy - nh / 2.0)
    nx = max(0, nx)
    ny = max(0, ny)
    nx2 = min(img_w, nx + nw)
    ny2 = min(img_h, ny + nh)
    return nx, ny, nx2 - nx, ny2 - ny


def get_annotation_summary(annotations: List[Dict]) -> Dict:
    """
    Get summary statistics from annotations.

    Args:
        annotations: List of frame annotations

    Returns:
        Summary dictionary
    """
    if not annotations:
        return {}

    predictions = [ann["prediction"] for ann in annotations]
    face_counts = [len(ann["faces"]) for ann in annotations]

    summary = {
        "total_frames": len(annotations),
        "frames_with_faces": sum(1 for ann in annotations if ann["faces"]),
        "average_confidence": float(np.mean(predictions)),
        "confidence_std": float(np.std(predictions)),
        "max_confidence": float(np.max(predictions)),
        "min_confidence": float(np.min(predictions)),
        "average_faces_per_frame": float(np.mean(face_counts)),
        "total_faces_detected": sum(face_counts),
        "fake_frames": sum(1 for p in predictions if p >= settings.THRESHOLD),
        "real_frames": sum(1 for p in predictions if p < settings.THRESHOLD),
    }

    return summary
