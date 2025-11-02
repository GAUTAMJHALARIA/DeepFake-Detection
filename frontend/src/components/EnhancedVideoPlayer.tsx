import React, { useState, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  IconButton,
  Slider,
  Card,
  CardContent,
  Chip,
} from '@mui/material';
import {
  PlayArrow,
  Pause,
  SkipNext,
  SkipPrevious,
  Speed,
  Fullscreen,
  VolumeUp,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { useHotkeys } from 'react-hotkeys-hook';

interface VideoFrame {
  index: number;
  timestamp: number;
  confidence: number;
  label: string;
  face_detected: boolean;
  confidence_color: [number, number, number];
  has_gradcam: boolean;
}

interface VideoInfo {
  duration: number;
  fps: number;
  total_frames: number;
  processed_frames: number;
  resolution: string;
  face_detect_rate: number;
  conversion_info?: {
    was_converted: boolean;
    conversion_message: string;
    original_path: string;
    final_path: string;
  };
}

interface EnhancedVideoPlayerProps {
  videoFile: File | null;
  analysisData: {
    id: string;
    score: number;
    label: string;
    video_info: VideoInfo;
    frames: VideoFrame[];
    statistics: any;
  };
  onFrameChange?: (frameIndex: number) => void;
  onTimeUpdate?: (currentTime: number) => void;
}

const EnhancedVideoPlayer: React.FC<EnhancedVideoPlayerProps> = ({
  videoFile,
  analysisData,
  onFrameChange,
  onTimeUpdate,
}) => {
  console.log('EnhancedVideoPlayer rendered with videoFile:', videoFile?.name, videoFile?.size, videoFile?.type);

  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [volume, setVolume] = useState(1);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [showConfidenceOverlay, setShowConfidenceOverlay] = useState(true);
  const [videoUrl, setVideoUrl] = useState<string>('');
  const [videoError, setVideoError] = useState<string>('');
  const [codecSupport, setCodecSupport] = useState<{[key: string]: boolean}>({});

  // Check browser codec support
  useEffect(() => {
    const checkCodecSupport = () => {
      const video = document.createElement('video');
      const codecs = {
        'mp4': 'video/mp4; codecs="avc1.42E01E"', // H.264 Baseline
        'mp4_h264': 'video/mp4; codecs="avc1.640028"', // H.264 High Profile
        'mp4_h265': 'video/mp4; codecs="hev1.1.6.L93.B0"', // H.265
        'webm': 'video/webm; codecs="vp8"',
        'webm_vp9': 'video/webm; codecs="vp9"',
        'ogg': 'video/ogg; codecs="theora"'
      };

      const support: {[key: string]: boolean} = {};
      Object.entries(codecs).forEach(([name, codec]) => {
        support[name] = video.canPlayType(codec) !== '';
      });

      console.log('Browser codec support:', support);
      setCodecSupport(support);
    };

    checkCodecSupport();
  }, []);

  // Create video URL from file or use converted video
  useEffect(() => {
    console.log('useEffect triggered for videoFile:', videoFile?.name, videoFile?.size, videoFile?.type);

    // For URL uploads, we need to fetch the video from the backend
    if (analysisData?.id && !videoFile) {
      console.log('No local video file, fetching from backend...');

      const fetchVideoFromBackend = async () => {
        try {
          console.log('Fetching video from backend:', analysisData.id);

          const response = await fetch(`http://localhost:8000/video/${analysisData.id}`, {
            method: 'GET',
            headers: {
              'Accept': 'video/mp4'
            }
          });

          console.log('Response status:', response.status);

          if (response.ok) {
            const blob = await response.blob();
            console.log('Video loaded successfully from backend:', blob.size, 'bytes');
            const url = URL.createObjectURL(blob);
            console.log('Video URL created:', url);
            setVideoUrl(url);

            return () => {
              URL.revokeObjectURL(url);
            };
          } else {
            const errorText = await response.text();
            console.error('Failed to load video from backend:', response.status, errorText);
            setVideoError('Video preview not available');
            return () => {};
          }
        } catch (error) {
          console.error('Error loading video from backend:', error);
          setVideoError('Failed to load video preview');
          return () => {};
        }
      };

      fetchVideoFromBackend();
    } else if (videoFile) {
      console.log('Creating video URL for file:', videoFile.name, videoFile.size, videoFile.type);
      setVideoError(''); // Clear any previous errors

      // Always try to fetch converted video from backend first (if analysis data exists)
      if (analysisData?.id) {
        console.log('Attempting to fetch converted video from backend...');

        // Try to fetch the converted video from the backend
        const fetchConvertedVideo = async () => {
          try {
            console.log('Fetching converted video from backend...');

            const response = await fetch(`http://localhost:8000/video/${analysisData.id}`, {
              method: 'GET',
              headers: {
                'Accept': 'video/mp4'
              }
            });

            console.log('Response status:', response.status);
            console.log('Response headers:', Object.fromEntries(response.headers.entries()));

            if (response.ok) {
              const blob = await response.blob();
              console.log('Blob size:', blob.size, 'Blob type:', blob.type);
              const url = URL.createObjectURL(blob);
              console.log('Converted video loaded successfully:', url);
              setVideoUrl(url);
              setVideoError(''); // Clear any errors

              return () => {
                URL.revokeObjectURL(url);
              };
            } else {
              const errorText = await response.text();
              console.warn('Failed to load converted video, status:', response.status, 'Error:', errorText);
              // Fall back to original video
              const url = URL.createObjectURL(videoFile);
              setVideoUrl(url);
              return () => URL.revokeObjectURL(url);
            }
          } catch (error) {
            console.error('Error loading converted video:', error);
            // Fall back to original video
      const url = URL.createObjectURL(videoFile);
      setVideoUrl(url);
      return () => URL.revokeObjectURL(url);
    }
        };

        fetchConvertedVideo();
        return;
      }

      // Validate file type
      if (!videoFile.type.startsWith('video/')) {
        console.error('Invalid file type:', videoFile.type);
        setVideoUrl('');
        setVideoError('Invalid file type. Please upload a video file.');
        return;
      }

      // Check if browser supports this video format
      const video = document.createElement('video');
      const canPlay = video.canPlayType(videoFile.type);
      console.log('Browser can play type:', videoFile.type, 'Result:', canPlay);

      if (canPlay === '') {
        console.error('Browser does not support this video format:', videoFile.type);
        setVideoUrl('');
        setVideoError(`Browser does not support this video format: ${videoFile.type}. The backend will automatically convert this video for you.`);
        return;
      }

      // Log warning for "maybe" results
      if (canPlay === 'maybe') {
        console.warn('Browser uncertain about video format support:', videoFile.type);
      }

      try {
        const url = URL.createObjectURL(videoFile);
        console.log('Video URL created successfully:', url);
        console.log('Video URL type:', typeof url);
        console.log('Video URL length:', url.length);
        setVideoUrl(url);

        return () => {
          console.log('Revoking video URL:', url);
          URL.revokeObjectURL(url);
        };
      } catch (error) {
        console.error('Failed to create video URL:', error);
        setVideoUrl('');
        setVideoError('Failed to load video file. Please try again.');
      }
    } else {
      console.log('No video file provided');
      setVideoUrl('');
    }
  }, [videoFile, analysisData]);

  // Handle uncaught video errors
  useEffect(() => {
    const handleUncaughtError = (event: ErrorEvent) => {
      if (event.error && event.error.name === 'NotSupportedError') {
        console.error('Uncaught NotSupportedError:', event.error);
        setVideoError('Video codec not supported. Your MP4 file likely uses H.265/HEVC or another unsupported codec. Please convert to H.264 Baseline MP4.');
      }
    };

    const handleUnhandledRejection = (event: PromiseRejectionEvent) => {
      if (event.reason && event.reason.name === 'NotSupportedError') {
        console.error('Unhandled NotSupportedError promise:', event.reason);
        setVideoError('Video codec not supported. Your MP4 file likely uses H.265/HEVC or another unsupported codec. Please convert to H.264 Baseline MP4.');
        event.preventDefault(); // Prevent the error from appearing in console
      }
    };

    window.addEventListener('error', handleUncaughtError);
    window.addEventListener('unhandledrejection', handleUnhandledRejection);

    return () => {
      window.removeEventListener('error', handleUncaughtError);
      window.removeEventListener('unhandledrejection', handleUnhandledRejection);
    };
  }, []);

  // Keyboard shortcuts
  useHotkeys('space', () => togglePlayPause(), [isPlaying]);
  useHotkeys('left', () => stepFrame(-1), [currentFrameIndex]);
  useHotkeys('right', () => stepFrame(1), [currentFrameIndex]);
  useHotkeys('up', () => changePlaybackRate(0.25), [playbackRate]);
  useHotkeys('down', () => changePlaybackRate(-0.25), [playbackRate]);

  const togglePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const stepFrame = (direction: number) => {
    if (videoRef.current && analysisData.frames.length > 0) {
      const newIndex = Math.max(0, Math.min(
        analysisData.frames.length - 1,
        currentFrameIndex + direction
      ));

      const targetTime = analysisData.frames[newIndex].timestamp;
      videoRef.current.currentTime = targetTime;
      setCurrentFrameIndex(newIndex);

      if (onFrameChange) {
        onFrameChange(newIndex);
      }
    }
  };

  const changePlaybackRate = (delta: number) => {
    const newRate = Math.max(0.25, Math.min(2, playbackRate + delta));
    setPlaybackRate(newRate);
    if (videoRef.current) {
      videoRef.current.playbackRate = newRate;
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      const time = videoRef.current.currentTime;
      setCurrentTime(time);

      // Find current frame based on timestamp
      const frameIndex = analysisData.frames.findIndex((frame, index) => {
        const nextFrame = analysisData.frames[index + 1];
        return time >= frame.timestamp && (!nextFrame || time < nextFrame.timestamp);
      });

      if (frameIndex !== -1 && frameIndex !== currentFrameIndex) {
        setCurrentFrameIndex(frameIndex);
        if (onFrameChange) {
          onFrameChange(frameIndex);
        }
      }

      if (onTimeUpdate) {
        onTimeUpdate(time);
      }
    }
  };

  const handleSeek = (value: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = value;
      setCurrentTime(value);
    }
  };

  const getCurrentFrame = () => {
    return analysisData.frames[currentFrameIndex] || null;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.7) return '#f44336'; // Red
    if (confidence >= 0.3) return '#ff9800'; // Orange
    return '#4caf50'; // Green
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getRecommendedFormats = () => {
    const recommendations = [];
    if (codecSupport.mp4 || codecSupport.mp4_h264) {
      recommendations.push('MP4 (H.264)');
    }
    if (codecSupport.webm || codecSupport.webm_vp9) {
      recommendations.push('WebM (VP8/VP9)');
    }
    if (codecSupport.ogg) {
      recommendations.push('OGG (Theora)');
    }
    return recommendations.length > 0 ? recommendations.join(', ') : 'No supported formats detected';
  };

  const getMP4ConversionGuide = () => {
    return {
      title: "MP4 Codec Issue Detected",
      description: "Your MP4 file likely uses H.265/HEVC codec which isn't widely supported in browsers.",
      solutions: [
        {
          title: "Convert to H.264 MP4 (Recommended)",
          command: "ffmpeg -i input.mp4 -c:v libx264 -profile:v baseline -c:a aac output.mp4",
          description: "This creates a browser-compatible MP4 file"
        },
        {
          title: "Convert to WebM (Alternative)",
          command: "ffmpeg -i input.mp4 -c:v libvpx-vp9 -c:a libopus output.webm",
          description: "WebM has excellent browser support"
        },
        {
          title: "Batch Convert Multiple Files",
          command: "for %f in (*.mp4) do ffmpeg -i \"%f\" -c:v libx264 -profile:v baseline -c:a aac \"converted_%f\"",
          description: "Convert all MP4 files in a folder"
        }
      ]
    };
  };

  const currentFrame = getCurrentFrame();

  // Safety check for hot reload issues
  if (!currentFrame && analysisData?.frames?.length > 0) {
    console.warn('currentFrame is null but frames exist, this might be a hot reload issue');
  }

  return (
    <Box>
      {/* Video Container */}
      <Paper sx={{ position: 'relative', mb: 2, overflow: 'hidden' }}>
        {videoUrl ? (
        <video
          ref={videoRef}
          src={videoUrl}
          style={{
            width: '100%',
            height: 'auto',
            maxHeight: '60vh',
            display: 'block',
          }}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={() => {
              console.log('Video metadata loaded');
              console.log('Video src:', videoRef.current?.src);
              console.log('Video currentSrc:', videoRef.current?.currentSrc);
            if (videoRef.current) {
              setDuration(videoRef.current.duration);
                console.log('Video duration:', videoRef.current.duration);
                console.log('Video dimensions:', videoRef.current.videoWidth, 'x', videoRef.current.videoHeight);
                console.log('Video ready state:', videoRef.current.readyState);
              }
            }}
            onPlay={() => {
              console.log('Video started playing');
              setIsPlaying(true);
            }}
            onPause={() => {
              console.log('Video paused');
              setIsPlaying(false);
            }}
            onError={(e) => {
              console.error('Video error:', e);
              console.error('Video error details:', {
                error: e.currentTarget.error,
                networkState: e.currentTarget.networkState,
                readyState: e.currentTarget.readyState,
                src: e.currentTarget.src,
                currentSrc: e.currentTarget.currentSrc
              });

              // Handle NotSupportedError specifically
              if (e.currentTarget.error && e.currentTarget.error.code === 4) {
                setVideoError('Video codec not supported. Your MP4 file likely uses H.265/HEVC or another unsupported codec. Please convert to H.264 Baseline MP4.');
                return;
              }

              // Set error message for display
              const error = e.currentTarget.error;
              if (error) {
                let errorMessage = 'Video playback error';
                switch (error.code) {
                  case error.MEDIA_ERR_ABORTED:
                    errorMessage = 'Video playback was aborted';
                    break;
                  case error.MEDIA_ERR_NETWORK:
                    errorMessage = 'Network error occurred while loading video';
                    break;
                  case error.MEDIA_ERR_DECODE:
                    errorMessage = 'Video decoding error - codec not supported. Your MP4 file may use an unsupported codec (like H.265/HEVC).';
                    break;
                  case error.MEDIA_ERR_SRC_NOT_SUPPORTED:
                    errorMessage = 'Video format not supported or source not found. Try converting to H.264 Baseline MP4.';
                    break;
                  default:
                    errorMessage = `Video error: ${error.message || 'Unknown error'}`;
                }

                // Add codec support information to the error
                const supportedFormats = Object.entries(codecSupport)
                  .filter(([_, supported]) => supported)
                  .map(([format, _]) => format)
                  .join(', ');

                if (supportedFormats) {
                  errorMessage += `\n\nSupported formats: ${supportedFormats}`;
                }

                setVideoError(errorMessage);
              }
            }}
            onLoadStart={() => {
              console.log('Video load started');
            }}
            onCanPlay={() => {
              console.log('Video can play');
            }}
            onLoadedData={() => {
              console.log('Video data loaded');
              setVideoError(''); // Clear any previous errors
            }}
          />
        ) : (
          <Box
            sx={{
              width: '100%',
              height: '300px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              backgroundColor: 'grey.100',
              color: 'grey.500',
            }}
          >
            <Typography variant="h6">
              {videoError ? videoError : (videoFile ? 'Loading video...' : 'No video file provided')}
            </Typography>
            {videoError && (
              <Box sx={{ mt: 2, textAlign: 'left', maxWidth: '600px' }}>
                <Typography variant="body2" sx={{ mb: 2 }}>
                  <strong>🎬 Video Processing Status:</strong> {analysisData?.video_info?.conversion_info?.conversion_message || 'Processing video compatibility...'}
                </Typography>

                <Typography variant="h6" sx={{ mb: 1, color: 'primary.main' }}>
                  ✨ What's Happening
                </Typography>

                <Box sx={{ mb: 2, p: 2, bgcolor: 'info.light', borderRadius: 1 }}>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>1. Upload Detection:</strong> Your MP4 file has been analyzed for browser compatibility
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>2. Backend Processing:</strong> Our system is ensuring optimal video format
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>3. Preview Ready:</strong> Video will be available for preview shortly
                  </Typography>
                </Box>

                <Typography variant="body2" sx={{ mt: 2, p: 1, bgcolor: 'success.light', borderRadius: 1 }}>
                  <strong>🚀 Smart Processing:</strong> Our backend automatically handles video compatibility issues.
                  Your video will work perfectly once processing is complete!
                </Typography>

                <Box sx={{ mt: 2, p: 2, bgcolor: 'warning.light', borderRadius: 1 }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 'bold', mb: 1 }}>
                    💡 Technical Details
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • Modern devices often record in H.265/HEVC for space efficiency
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    • Browsers prefer H.264 for maximum compatibility
                  </Typography>
                  <Typography variant="body2">
                    • Our system automatically converts when needed
                  </Typography>
                </Box>
              </Box>
            )}
            {videoUrl && (
              <Typography variant="caption" sx={{ mt: 1, wordBreak: 'break-all' }}>
                Video URL: {videoUrl}
              </Typography>
            )}
            {videoFile && !videoUrl && (
              <Typography variant="caption" sx={{ mt: 1 }}>
                File: {videoFile.name} ({videoFile.size} bytes, {videoFile.type})
              </Typography>
            )}
            <Box sx={{ mt: 2 }}>
              <Typography variant="caption" color="textSecondary">
                Browser Codec Support:
              </Typography>
              <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {Object.entries(codecSupport).map(([format, supported]) => (
                  <Chip
                    key={format}
                    label={format}
                    size="small"
                    color={supported ? 'success' : 'error'}
                    variant={supported ? 'filled' : 'outlined'}
                  />
                ))}
              </Box>
            </Box>
          </Box>
        )}

        {/* Confidence Overlay */}
        {showConfidenceOverlay && currentFrame && (
          <Box
            sx={{
              position: 'absolute',
              top: 16,
              right: 16,
              zIndex: 10,
            }}
          >
            <Chip
              label={`${((currentFrame?.confidence || 0) * 100).toFixed(1)}% ${currentFrame?.label || 'Unknown'}`}
              sx={{
                backgroundColor: getConfidenceColor(currentFrame?.confidence || 0),
                color: 'white',
                fontWeight: 'bold',
                fontSize: '1rem',
                padding: '8px 16px',
              }}
            />
          </Box>
        )}

        {/* Frame Info Overlay */}
        <Box
          sx={{
            position: 'absolute',
            bottom: 16,
            left: 16,
            backgroundColor: 'rgba(0, 0, 0, 0.7)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: 1,
            fontSize: '0.875rem',
          }}
        >
          Frame {currentFrameIndex + 1} / {analysisData.frames.length} | {formatTime(currentTime)}
        </Box>
      </Paper>

      {/* Timeline with Confidence Visualization */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Timeline & Confidence
        </Typography>

        {/* Custom Timeline */}
        <Box sx={{ position: 'relative', height: 60, mb: 2 }}>
          {/* Confidence Background */}
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              height: 40,
              background: `linear-gradient(to right, ${analysisData.frames.map((frame, index) => {
                const position = (index / (analysisData.frames.length - 1)) * 100;
                const color = getConfidenceColor(frame.confidence);
                return `${color} ${position}%`;
              }).join(', ')})`,
              borderRadius: 1,
              cursor: 'pointer',
            }}
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const clickX = e.clientX - rect.left;
              const percentage = clickX / rect.width;
              const targetTime = percentage * duration;
              handleSeek(targetTime);
            }}
          />

          {/* Current Position Indicator */}
          <Box
            sx={{
              position: 'absolute',
              top: 0,
              left: `${(currentTime / duration) * 100}%`,
              width: 3,
              height: 40,
              backgroundColor: 'white',
              boxShadow: '0 0 4px rgba(0,0,0,0.5)',
              transform: 'translateX(-50%)',
            }}
          />

          {/* Time Labels */}
          <Box
            sx={{
              position: 'absolute',
              top: 45,
              left: 0,
              right: 0,
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: '0.75rem',
              color: 'text.secondary',
            }}
          >
            <span>0:00</span>
            <span>{formatTime(duration)}</span>
          </Box>
        </Box>

        {/* Standard Slider for Fine Control */}
        <Slider
          value={currentTime}
          max={duration}
          onChange={(_, value) => handleSeek(value as number)}
          sx={{ mb: 1 }}
        />
      </Paper>

      {/* Controls */}
      <Paper sx={{ p: 2 }}>
        <Box display="flex" alignItems="center" gap={2} flexWrap="wrap">
          {/* Playback Controls */}
          <Box display="flex" alignItems="center" gap={1}>
            <IconButton onClick={() => stepFrame(-1)} disabled={currentFrameIndex === 0}>
              <SkipPrevious />
            </IconButton>

            <IconButton onClick={togglePlayPause} size="large">
              {isPlaying ? <Pause /> : <PlayArrow />}
            </IconButton>

            <IconButton
              onClick={() => stepFrame(1)}
              disabled={currentFrameIndex === analysisData.frames.length - 1}
            >
              <SkipNext />
            </IconButton>
          </Box>

          {/* Speed Control */}
          <Box display="flex" alignItems="center" gap={1}>
            <Speed />
            <Typography variant="body2" sx={{ minWidth: 40 }}>
              {playbackRate}x
            </Typography>
            <Slider
              value={playbackRate}
              min={0.25}
              max={2}
              step={0.25}
              onChange={(_, value) => changePlaybackRate((value as number) - playbackRate)}
              sx={{ width: 100 }}
            />
          </Box>

          {/* Volume Control */}
          <Box display="flex" alignItems="center" gap={1}>
            <VolumeUp />
            <Slider
              value={volume}
              min={0}
              max={1}
              step={0.1}
              onChange={(_, value) => {
                setVolume(value as number);
                if (videoRef.current) {
                  videoRef.current.volume = value as number;
                }
              }}
              sx={{ width: 100 }}
            />
          </Box>

          {/* Toggle Overlay */}
          <Chip
            label="Confidence Overlay"
            variant={showConfidenceOverlay ? "filled" : "outlined"}
            onClick={() => setShowConfidenceOverlay(!showConfidenceOverlay)}
            clickable
          />
        </Box>

        {/* Keyboard Shortcuts Help */}
        <Box mt={2}>
          <Typography variant="caption" color="textSecondary">
            Shortcuts: Space (play/pause) | ←/→ (frame step) | ↑/↓ (speed)
          </Typography>
        </Box>
      </Paper>

      {/* Current Frame Details */}
      {currentFrame && (
        <Card sx={{ mt: 2 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Current Frame Details
            </Typography>

            <Box display="flex" gap={4} flexWrap="wrap">
              <Box>
                <Typography variant="body2" color="textSecondary">
                  Confidence
                </Typography>
                <Typography variant="h6" color={getConfidenceColor(currentFrame?.confidence || 0)}>
                  {((currentFrame?.confidence || 0) * 100).toFixed(1)}%
                </Typography>
              </Box>

              <Box>
                <Typography variant="body2" color="textSecondary">
                  Classification
                </Typography>
                <Typography variant="h6">
                  {(currentFrame?.label || 'Unknown').toUpperCase()}
                </Typography>
              </Box>

              <Box>
                <Typography variant="body2" color="textSecondary">
                  Face Detected
                </Typography>
                <Typography variant="h6">
                  {currentFrame?.face_detected ? 'Yes' : 'No'}
                </Typography>
              </Box>

              <Box>
                <Typography variant="body2" color="textSecondary">
                  Timestamp
                </Typography>
                <Typography variant="h6">
                  {(currentFrame?.timestamp || 0).toFixed(2)}s
                </Typography>
              </Box>

              {currentFrame?.has_gradcam && (
                <Box>
                  <Typography variant="body2" color="textSecondary">
                    Explainability
                  </Typography>
                  <Chip label="Grad-CAM Available" color="info" size="small" />
                </Box>
              )}
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default EnhancedVideoPlayer;
