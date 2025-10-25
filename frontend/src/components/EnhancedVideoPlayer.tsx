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
}

interface EnhancedVideoPlayerProps {
  videoFile: File;
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
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [volume, setVolume] = useState(1);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [showConfidenceOverlay, setShowConfidenceOverlay] = useState(true);
  const [videoUrl, setVideoUrl] = useState<string>('');

  // Create video URL from file
  useEffect(() => {
    if (videoFile) {
      const url = URL.createObjectURL(videoFile);
      setVideoUrl(url);
      return () => URL.revokeObjectURL(url);
    }
  }, [videoFile]);

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

  const currentFrame = getCurrentFrame();

  return (
    <Box>
      {/* Video Container */}
      <Paper sx={{ position: 'relative', mb: 2, overflow: 'hidden' }}>
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
            if (videoRef.current) {
              setDuration(videoRef.current.duration);
            }
          }}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
        />

        {/* Confidence Overlay */}
        {showConfidenceOverlay && currentFrame && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            style={{
              position: 'absolute',
              top: 16,
              right: 16,
              zIndex: 10,
            }}
          >
            <Chip
              label={`${(currentFrame.confidence * 100).toFixed(1)}% ${currentFrame.label}`}
              sx={{
                backgroundColor: getConfidenceColor(currentFrame.confidence),
                color: 'white',
                fontWeight: 'bold',
                fontSize: '1rem',
                padding: '8px 16px',
              }}
            />
          </motion.div>
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
                <Typography variant="h6" color={getConfidenceColor(currentFrame.confidence)}>
                  {(currentFrame.confidence * 100).toFixed(1)}%
                </Typography>
              </Box>

              <Box>
                <Typography variant="body2" color="textSecondary">
                  Classification
                </Typography>
                <Typography variant="h6">
                  {currentFrame.label.toUpperCase()}
                </Typography>
              </Box>

              <Box>
                <Typography variant="body2" color="textSecondary">
                  Face Detected
                </Typography>
                <Typography variant="h6">
                  {currentFrame.face_detected ? 'Yes' : 'No'}
                </Typography>
              </Box>

              <Box>
                <Typography variant="body2" color="textSecondary">
                  Timestamp
                </Typography>
                <Typography variant="h6">
                  {currentFrame.timestamp.toFixed(2)}s
                </Typography>
              </Box>

              {currentFrame.has_gradcam && (
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
