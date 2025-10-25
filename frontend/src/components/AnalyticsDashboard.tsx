import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
} from '@mui/material';

interface AnalyticsDashboardProps {
  statistics: {
    mean_confidence: number;
    confidence_variance: number;
    max_confidence: number;
    min_confidence: number;
    suspicious_frames: number;
    total_frames: number;
    quality_score: number;
  };
  videoInfo: {
    duration: number;
    fps: number;
    total_frames: number;
    processed_frames: number;
    resolution: string;
    face_detect_rate: number;
  };
  processingInfo: {
    gradcam_enabled: boolean;
    all_frames_extracted: boolean;
    max_resolution: string;
    threshold: number;
  };
  latencyMs: number;
}

const AnalyticsDashboard: React.FC<AnalyticsDashboardProps> = ({
  statistics,
  videoInfo,
  processingInfo,
  latencyMs,
}) => {
  return (
    <Box sx={{ flexGrow: 1, p: 2 }}>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Confidence Statistics
              </Typography>
              <Typography>Mean Confidence: {statistics.mean_confidence.toFixed(3)}</Typography>
              <Typography>Variance: {statistics.confidence_variance.toFixed(3)}</Typography>
              <Typography>Max Confidence: {statistics.max_confidence.toFixed(3)}</Typography>
              <Typography>Min Confidence: {statistics.min_confidence.toFixed(3)}</Typography>
              <Typography>Suspicious Frames: {statistics.suspicious_frames}</Typography>
              <Typography>Quality Score: {statistics.quality_score.toFixed(2)}</Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Video Information
              </Typography>
              <Typography>Duration: {videoInfo.duration.toFixed(2)}s</Typography>
              <Typography>FPS: {videoInfo.fps}</Typography>
              <Typography>Resolution: {videoInfo.resolution}</Typography>
              <Typography>Total Frames: {videoInfo.total_frames}</Typography>
              <Typography>Processed Frames: {videoInfo.processed_frames}</Typography>
              <Typography>Face Detection Rate: {(videoInfo.face_detect_rate * 100).toFixed(1)}%</Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Processing Information
              </Typography>
              <Typography>GradCAM Enabled: {processingInfo.gradcam_enabled ? 'Yes' : 'No'}</Typography>
              <Typography>All Frames Extracted: {processingInfo.all_frames_extracted ? 'Yes' : 'No'}</Typography>
              <Typography>Max Resolution: {processingInfo.max_resolution}</Typography>
              <Typography>Threshold: {processingInfo.threshold}</Typography>
              <Typography>Processing Latency: {latencyMs}ms</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default AnalyticsDashboard;
