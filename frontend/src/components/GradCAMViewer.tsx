import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Card,
  CardContent,
  Button,
  LinearProgress,
  Alert,
  Slider,
  FormControlLabel,
  Switch,
  Grid,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Visibility,
  Download,
  Refresh,
  NavigateBefore,
  NavigateNext,
  ZoomIn,
  ZoomOut,
} from '@mui/icons-material';
import axios from 'axios';

interface GradCAMViewerProps {
  analysisId: string;
  currentFrameIndex: number;
  totalFrames: number;
  onFrameChange: (frameIndex: number) => void;
}

interface GradCAMData {
  frame_index: number;
  heatmap_base64: string;
  overlay_base64?: string;
}

const GradCAMViewer: React.FC<GradCAMViewerProps> = ({
  analysisId,
  currentFrameIndex,
  totalFrames,
  onFrameChange,
}) => {
  const [gradcamData, setGradcamData] = useState<GradCAMData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [opacity, setOpacity] = useState(0.6);
  const [zoom, setZoom] = useState(2);
  const [frameData, setFrameData] = useState<any>(null);

  // Load Grad-CAM data for current frame
  const loadGradCAM = async (frameIndex: number) => {
    setLoading(true);
    setError(null);

    try {
      // Load Grad-CAM heatmap
      const gradcamResponse = await axios.get(
        `http://localhost:8000/gradcam/${analysisId}/${frameIndex}`,
        {
          headers: {
            'Authorization': 'Bearer change-me',
          },
        }
      );

      // Load frame data
      const frameResponse = await axios.get(
        `http://localhost:8000/frames/${analysisId}/${frameIndex}`,
        {
          headers: {
            'Authorization': 'Bearer change-me',
          },
        }
      );

      setGradcamData(gradcamResponse.data);
      setFrameData(frameResponse.data);

    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load Grad-CAM data');
      setGradcamData(null);
      setFrameData(null);
    } finally {
      setLoading(false);
    }
  };

  // Load data when frame changes
  useEffect(() => {
    loadGradCAM(currentFrameIndex);
  }, [currentFrameIndex, analysisId]);

  const handleFrameNavigation = (direction: number) => {
    const newIndex = Math.max(0, Math.min(totalFrames - 1, currentFrameIndex + direction));
    onFrameChange(newIndex);
  };

  const downloadHeatmap = () => {
    if (!gradcamData) return;

    const link = document.createElement('a');
    link.href = `data:image/png;base64,${gradcamData.heatmap_base64}`;
    link.download = `gradcam_frame_${currentFrameIndex + 1}.png`;
    link.click();
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.7) return '#f44336'; // Red
    if (confidence >= 0.3) return '#ff9800'; // Orange
    return '#4caf50'; // Green
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Explainable AI - Grad-CAM++ Visualization
      </Typography>
      <Typography variant="body2" color="textSecondary" paragraph>
        Visual explanation of model decisions using Gradient-weighted Class Activation Mapping.
        Red areas indicate regions that most influence the deepfake classification.
      </Typography>

      {/* Controls */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={4}>
              <Box display="flex" alignItems="center" gap={1}>
                <IconButton
                  onClick={() => handleFrameNavigation(-1)}
                  disabled={currentFrameIndex === 0}
                >
                  <NavigateBefore />
                </IconButton>

                <Typography variant="body2" sx={{ minWidth: 120, textAlign: 'center' }}>
                  Frame {currentFrameIndex + 1} / {totalFrames}
                </Typography>

                <IconButton
                  onClick={() => handleFrameNavigation(1)}
                  disabled={currentFrameIndex === totalFrames - 1}
                >
                  <NavigateNext />
                </IconButton>
              </Box>
            </Grid>

            <Grid item xs={12} md={4}>
              <Box display="flex" alignItems="center" gap={2}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={showOverlay}
                      onChange={(e) => setShowOverlay(e.target.checked)}
                    />
                  }
                  label="Show Overlay"
                />

                <Tooltip title="Refresh Grad-CAM">
                  <IconButton onClick={() => loadGradCAM(currentFrameIndex)}>
                    <Refresh />
                  </IconButton>
                </Tooltip>
              </Box>
            </Grid>

            <Grid item xs={12} md={4}>
              <Box display="flex" alignItems="center" gap={1}>
                <Button
                  variant="outlined"
                  startIcon={<Download />}
                  onClick={downloadHeatmap}
                  disabled={!gradcamData}
                  size="small"
                >
                  Download
                </Button>
              </Box>
            </Grid>
          </Grid>

          {/* Opacity and Zoom Controls */}
          <Box mt={2}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Typography variant="body2" gutterBottom>
                  Heatmap Opacity: {Math.round(opacity * 100)}%
                </Typography>
                <Slider
                  value={opacity}
                  onChange={(_, value) => setOpacity(value as number)}
                  min={0.1}
                  max={1}
                  step={0.1}
                  disabled={!showOverlay}
                />
              </Grid>

              <Grid item xs={12} md={6}>
                <Typography variant="body2" gutterBottom>
                  Zoom: {Math.round(zoom * 100)}%
                </Typography>
                <Box display="flex" alignItems="center" gap={1}>
                  <IconButton size="small" onClick={() => setZoom(Math.max(0.5, zoom - 0.25))}>
                    <ZoomOut />
                  </IconButton>
                  <Slider
                    value={zoom}
                    onChange={(_, value) => setZoom(value as number)}
                    min={0.5}
                    max={4}
                    step={0.25}
                    sx={{ flexGrow: 1 }}
                  />
                  <IconButton size="small" onClick={() => setZoom(Math.min(4, zoom + 0.25))}>
                    <ZoomIn />
                  </IconButton>
                </Box>
              </Grid>
            </Grid>
          </Box>
        </CardContent>
      </Card>

      {/* Loading State */}
      {loading && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="body2" gutterBottom>
              Loading Grad-CAM visualization...
            </Typography>
            <LinearProgress />
          </CardContent>
        </Card>
      )}

      {/* Error State */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Grad-CAM Visualization */}
      {gradcamData && !loading && (
        <Grid container spacing={3}>
          {/* Main Visualization */}
          <Grid item xs={12} lg={8}>
            <Paper sx={{ p: 2, textAlign: 'center' }}>
              <Typography variant="h6" gutterBottom>
                Attention Heatmap
              </Typography>

              <Box
                sx={{
                  position: 'relative',
                  display: 'inline-block',
                  border: '1px solid',
                  borderColor: 'divider',
                  borderRadius: 1,
                  overflow: 'hidden',
                }}
              >
                {/* Original Frame (if available) */}
                {frameData?.thumbnail_base64 && (
                  <img
                    src={`data:image/png;base64,${frameData.thumbnail_base64}`}
                    alt="Original frame"
                    style={{
                      display: 'block',
                      width: '800px',
                      height: 'auto',
                      minWidth: '600px',
                      transform: `scale(${zoom})`,
                      transformOrigin: 'center',
                    }}
                  />
                )}

                {/* Grad-CAM Overlay */}
                {showOverlay && (
                  <img
                    src={`data:image/png;base64,${gradcamData.heatmap_base64}`}
                    alt="Grad-CAM heatmap"
                    style={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: '100%',
                      opacity: opacity,
                      mixBlendMode: 'multiply',
                      transform: `scale(${zoom})`,
                      transformOrigin: 'center',
                    }}
                  />
                )}
              </Box>

              <Typography variant="caption" color="textSecondary" display="block" sx={{ mt: 1 }}>
                Red regions indicate areas that most influence the deepfake classification
              </Typography>
            </Paper>
          </Grid>

          {/* Frame Information */}
          <Grid item xs={12} lg={4}>
            <Paper sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Frame Analysis
              </Typography>

              {frameData && (
                <Box>
                  <Box mb={2}>
                    <Typography variant="body2" color="textSecondary">
                      Confidence Score
                    </Typography>
                    <Box display="flex" alignItems="center" gap={2}>
                      <LinearProgress
                        variant="determinate"
                        value={frameData.confidence * 100}
                        sx={{
                          flexGrow: 1,
                          height: 8,
                          borderRadius: 4,
                          '& .MuiLinearProgress-bar': {
                            backgroundColor: getConfidenceColor(frameData.confidence),
                          },
                        }}
                      />
                      <Typography
                        variant="h6"
                        sx={{ color: getConfidenceColor(frameData.confidence) }}
                      >
                        {(frameData.confidence * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  </Box>

                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="textSecondary">
                        Classification
                      </Typography>
                      <Typography variant="body1" fontWeight="bold">
                        {frameData.label.toUpperCase()}
                      </Typography>
                    </Grid>

                    <Grid item xs={6}>
                      <Typography variant="body2" color="textSecondary">
                        Face Detected
                      </Typography>
                      <Typography variant="body1">
                        {frameData.face_detected ? 'Yes' : 'No'}
                      </Typography>
                    </Grid>

                    <Grid item xs={12}>
                      <Typography variant="body2" color="textSecondary">
                        Timestamp
                      </Typography>
                      <Typography variant="body1">
                        {frameData.timestamp.toFixed(2)}s
                      </Typography>
                    </Grid>
                  </Grid>
                </Box>
              )}
            </Paper>

            {/* Interpretation Guide */}
            <Paper sx={{ p: 2, mt: 2 }}>
              <Typography variant="h6" gutterBottom>
                Interpretation Guide
              </Typography>

              <Box mb={2}>
                <Typography variant="body2" fontWeight="bold" gutterBottom>
                  Color Coding:
                </Typography>
                <Box display="flex" alignItems="center" gap={1} mb={1}>
                  <Box width={16} height={16} bgcolor="#ff0000" borderRadius="50%" />
                  <Typography variant="body2">High attention (most important)</Typography>
                </Box>
                <Box display="flex" alignItems="center" gap={1} mb={1}>
                  <Box width={16} height={16} bgcolor="#ffff00" borderRadius="50%" />
                  <Typography variant="body2">Medium attention</Typography>
                </Box>
                <Box display="flex" alignItems="center" gap={1}>
                  <Box width={16} height={16} bgcolor="#0000ff" borderRadius="50%" />
                  <Typography variant="body2">Low attention</Typography>
                </Box>
              </Box>

              <Typography variant="body2" color="textSecondary">
                <strong>What this shows:</strong> The model focuses on specific facial regions
                when making its decision. Red areas are the most influential in determining
                whether the frame contains deepfake artifacts.
              </Typography>
            </Paper>

            {/* Technical Details */}
            <Paper sx={{ p: 2, mt: 2 }}>
              <Typography variant="h6" gutterBottom>
                Technical Details
              </Typography>

              <Typography variant="body2" color="textSecondary" paragraph>
                <strong>Method:</strong> Grad-CAM++ (Gradient-weighted Class Activation Mapping)
              </Typography>

              <Typography variant="body2" color="textSecondary" paragraph>
                <strong>Purpose:</strong> Provides visual explanations for CNN-based deepfake
                detection by highlighting important regions in the input image.
              </Typography>

              <Typography variant="body2" color="textSecondary">
                <strong>Reliability:</strong> Higher confidence scores generally correlate
                with more focused attention patterns.
              </Typography>
            </Paper>
          </Grid>
        </Grid>
      )}

      {/* No Data State */}
      {!gradcamData && !loading && !error && (
        <Paper sx={{ p: 4, textAlign: 'center' }}>
          <Visibility sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
          <Typography variant="h6" color="textSecondary" gutterBottom>
            No Grad-CAM Data Available
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Grad-CAM visualization is not available for this frame.
            This might be due to processing limitations or the frame not containing detectable faces.
          </Typography>
        </Paper>
      )}
    </Box>
  );
};

export default GradCAMViewer;
