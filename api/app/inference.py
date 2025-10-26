"""
Simplified inference module for deepfake detection.
Contains only the essential functions needed for the enhanced analysis system.
"""


def get_supported_formats() -> dict:
    """
    Get list of supported video and image formats.

    Returns:
        Dictionary with supported formats
    """
    return {
        "video_formats": ["mp4", "avi", "mov", "mkv", "wmv", "flv", "webm", "m4v"],
        "image_formats": ["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        "max_file_size_mb": 100,
        "max_resolution": "1920x1080",
        "min_resolution": "64x64",
    }
