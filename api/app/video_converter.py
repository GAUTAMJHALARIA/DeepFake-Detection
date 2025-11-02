import os
import tempfile
import logging
from pathlib import Path
from typing import Optional, Tuple
import ffmpeg

logger = logging.getLogger(__name__)


class VideoConverter:
    """Handles video conversion for browser compatibility"""

    def __init__(self):
        self.temp_dir = Path(tempfile.gettempdir()) / "dfd_conversions"
        self.temp_dir.mkdir(exist_ok=True)

    def check_codec_support(self, video_path: str) -> Tuple[bool, str]:
        """
        Check if video codec is browser-compatible
        Returns: (is_compatible, codec_info)
        """
        try:
            # First check if FFmpeg is available
            try:
                ffmpeg.probe(video_path)
            except ffmpeg.Error as e:
                logger.error(f"FFmpeg not available or error: {e}")
                return False, "FFmpeg not available"

            probe = ffmpeg.probe(video_path)
            video_stream = next(
                (
                    stream
                    for stream in probe["streams"]
                    if stream["codec_type"] == "video"
                ),
                None,
            )

            if not video_stream:
                return False, "No video stream found"

            codec_name = video_stream.get("codec_name", "unknown")
            profile = video_stream.get("profile", "")

            # Check for browser-compatible codecs
            compatible_codecs = ["h264", "vp8", "vp9", "theora"]
            is_compatible = codec_name.lower() in compatible_codecs

            # Additional check for H.264 profile
            if codec_name.lower() == "h264":
                # High profile H.264 might not work in all browsers
                if profile and "high" in profile.lower():
                    is_compatible = False
                    codec_info = "H.264 High Profile (not browser-compatible)"
                else:
                    codec_info = f"H.264 {profile or 'Baseline'} Profile"
            else:
                codec_info = f"{codec_name.upper()} {profile or ''}".strip()

            logger.info(f"Codec check: {codec_info}, compatible: {is_compatible}")
            return is_compatible, codec_info

        except Exception as e:
            logger.error(f"Error checking codec support: {e}")
            return False, f"Error: {str(e)}"

    def convert_to_browser_compatible(
        self, input_path: str, output_path: Optional[str] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Convert video to browser-compatible format
        Returns: (success, message, output_path)
        """
        try:
            # Determine output path
            if not output_path:
                # Generate output path
                input_file = Path(input_path)
                output_path = str(self.temp_dir / f"converted_{input_file.stem}.mp4")
            else:
                # Ensure it's a string
                output_path = str(output_path)

            # Convert to H.264 Baseline Profile for maximum browser compatibility
            (
                ffmpeg.input(input_path)
                .output(
                    output_path,
                    vcodec="libx264",
                    acodec="aac",
                    vprofile="baseline",
                    preset="fast",
                    crf=23,  # Good quality
                    movflags="faststart",  # Enable progressive download
                )
                .overwrite_output()
                .run(quiet=True)
            )

            # Verify the conversion was successful
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Successfully converted {input_path} to {output_path}")
                return True, "Video converted successfully", output_path
            else:
                return (
                    False,
                    "Conversion failed - output file is empty or missing",
                    None,
                )

        except ffmpeg.Error as e:
            error_msg = f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}"
            logger.error(error_msg)
            return False, error_msg, None
        except Exception as e:
            error_msg = f"Conversion error: {str(e)}"
            logger.error(error_msg)
            return False, error_msg, None

    def get_video_info(self, video_path: str) -> dict:
        """Get detailed video information"""
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next(
                (
                    stream
                    for stream in probe["streams"]
                    if stream["codec_type"] == "video"
                ),
                None,
            )
            audio_stream = next(
                (
                    stream
                    for stream in probe["streams"]
                    if stream["codec_type"] == "audio"
                ),
                None,
            )

            info = {
                "duration": float(probe["format"].get("duration", 0)),
                "size": int(probe["format"].get("size", 0)),
                "bitrate": int(probe["format"].get("bit_rate", 0)),
            }

            if video_stream:
                info.update(
                    {
                        "width": int(video_stream.get("width", 0)),
                        "height": int(video_stream.get("height", 0)),
                        "fps": eval(video_stream.get("r_frame_rate", "0/1")),
                        "video_codec": video_stream.get("codec_name", "unknown"),
                        "video_profile": video_stream.get("profile", ""),
                    }
                )

            if audio_stream:
                info.update(
                    {
                        "audio_codec": audio_stream.get("codec_name", "unknown"),
                        "audio_channels": int(audio_stream.get("channels", 0)),
                        "audio_sample_rate": int(audio_stream.get("sample_rate", 0)),
                    }
                )

            return info

        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {"error": str(e)}

    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up old temporary conversion files"""
        try:
            import time

            current_time = time.time()
            max_age_seconds = max_age_hours * 3600

            for file_path in self.temp_dir.glob("converted_*"):
                if current_time - file_path.stat().st_mtime > max_age_seconds:
                    file_path.unlink()
                    logger.info(f"Cleaned up old conversion file: {file_path}")

        except Exception as e:
            logger.error(f"Error cleaning up temp files: {e}")


# Global converter instance
video_converter = VideoConverter()
