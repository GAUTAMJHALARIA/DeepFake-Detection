import React from 'react';
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  LinearProgress,
  CircularProgress,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Timeline,
  Speed,
  Security,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

interface EnhancedAnalyticsDashboardProps {
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

interface StatCardProps {
  icon: React.ReactNode;
  title: string;
  value: string | number;
  subtitle: string;
  color: string;
}

const StatCard: React.FC<StatCardProps> = ({ icon, title, value, subtitle, color }) => (
  <motion.div
    whileHover={{ scale: 1.05, y: -4 }}
    transition={{ duration: 0.2 }}
  >
    <Card sx={{
      height: '100%',
      background: `linear-gradient(135deg, ${color}15 0%, ${color}05 100%)`,
      border: `1px solid ${color}30`
    }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
          {icon}
          <Typography variant="h3" color={color} fontWeight="bold">
            {value}
          </Typography>
        </Box>
        <Typography variant="h6" gutterBottom fontWeight="600">
          {title}
        </Typography>
        <Typography variant="caption" color="textSecondary">
          {subtitle}
        </Typography>
      </CardContent>
    </Card>
  </motion.div>
);

interface CircularProgressCardProps {
  label: string;
  value: number;
  color: string;
  icon: React.ReactNode;
}

const CircularProgressCard: React.FC<CircularProgressCardProps> = ({
  label,
  value,
  color,
  icon
}) => (
  <Card sx={{ height: '100%' }}>
    <CardContent>
      <Box display="flex" alignItems="center" justifyContent="center" flexDirection="column">
        <Box position="relative" display="inline-flex" mb={2}>
          <CircularProgress
            variant="determinate"
            value={value * 100}
            size={120}
            thickness={4}
            sx={{ color }}
          />
          <Box
            sx={{
              top: 0,
              left: 0,
              bottom: 0,
              right: 0,
              position: 'absolute',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Typography variant="h4" component="div" color={color} fontWeight="bold">
              {Math.round(value * 100)}
            </Typography>
          </Box>
        </Box>
        <Box mb={1}>{icon}</Box>
        <Typography variant="h6" mt={1} fontWeight="600">
          {label}
        </Typography>
      </Box>
    </CardContent>
  </Card>
);

const EnhancedAnalyticsDashboard: React.FC<EnhancedAnalyticsDashboardProps> = ({
  statistics,
  videoInfo,
  processingInfo,
  latencyMs,
}) => {
  const getColorByValue = (value: number, threshold: number): 'error' | 'warning' | 'success' => {
    if (value > threshold) return 'error';
    if (value > threshold * 0.5) return 'warning';
    return 'success';
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold', mb: 3 }}>
        📈 Deep Analytics Dashboard
      </Typography>

      <Grid container spacing={3}>
        {/* Key Metrics */}
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            icon={<TrendingUp sx={{ fontSize: 40, color: 'primary.main' }} />}
            title="Detection Confidence"
            value={`${(statistics.mean_confidence * 100).toFixed(1)}%`}
            subtitle="Average across all frames"
            color="primary.main"
          />
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            icon={<TrendingDown sx={{ fontSize: 40, color: 'error.main' }} />}
            title="Suspicious Frames"
            value={statistics.suspicious_frames}
            subtitle={`of ${statistics.total_frames} total`}
            color="error.main"
          />
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            icon={<Timeline sx={{ fontSize: 40, color: 'info.main' }} />}
            title="Face Detection Rate"
            value={`${(videoInfo.face_detect_rate * 100).toFixed(1)}%`}
            subtitle="Frames with faces detected"
            color="info.main"
          />
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            icon={<Speed sx={{ fontSize: 40, color: 'success.main' }} />}
            title="Processing Speed"
            value={`${(latencyMs / 1000).toFixed(1)}s`}
            subtitle="Total analysis time"
            color="success.main"
          />
        </Grid>

        {/* Quality Metrics */}
        <Grid item xs={12} md={4}>
          <CircularProgressCard
            label="Overall Quality Score"
            value={statistics.quality_score}
            color="#4caf50"
            icon={<Security sx={{ fontSize: 32, color: '#4caf50' }} />}
          />
        </Grid>

        <Grid item xs={12} md={4}>
          <CircularProgressCard
            label="Detection Confidence"
            value={statistics.mean_confidence}
            color={
              getColorByValue(statistics.mean_confidence, 0.7) === 'error' ? '#f44336' :
              getColorByValue(statistics.mean_confidence, 0.7) === 'warning' ? '#ff9800' : '#4caf50'
            }
            icon={<TrendingUp sx={{ fontSize: 32 }} />}
          />
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom fontWeight="600">
                📊 Confidence Distribution
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="caption" fontWeight="600">Real Confidence</Typography>
                  <Typography variant="caption" fontWeight="600">
                    {Math.round((1 - statistics.mean_confidence) * 100)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={(1 - statistics.mean_confidence) * 100}
                  sx={{
                    height: 10,
                    borderRadius: 2,
                    bgcolor: 'success.light'
                  }}
                  color="success"
                />
              </Box>
              <Box>
                <Box display="flex" justifyContent="space-between" mb={1}>
                  <Typography variant="caption" fontWeight="600">Fake Confidence</Typography>
                  <Typography variant="caption" fontWeight="600">
                    {Math.round(statistics.mean_confidence * 100)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={statistics.mean_confidence * 100}
                  sx={{
                    height: 10,
                    borderRadius: 2,
                    bgcolor: 'error.light'
                  }}
                  color={getColorByValue(statistics.mean_confidence, 0.5)}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Variance & Range */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom fontWeight="600">
                📊 Confidence Variance
              </Typography>
              <Typography variant="h2" color="info.main" fontWeight="bold">
                {statistics.confidence_variance.toFixed(3)}
              </Typography>
              <Typography variant="body2" color="textSecondary" mt={1}>
                Higher variance indicates inconsistent deepfake artifacts
              </Typography>
              <Box mt={2}>
                <Typography variant="caption" color="textSecondary">
                  Range: {statistics.min_confidence.toFixed(3)} - {statistics.max_confidence.toFixed(3)}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom fontWeight="600">
                ⚙️ Processing Details
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2">Grad-CAM:</Typography>
                  <Typography fontWeight="bold" color={processingInfo.gradcam_enabled ? 'success.main' : 'text.secondary'}>
                    {processingInfo.gradcam_enabled ? '✓ Enabled' : 'Disabled'}
                  </Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2">Max Resolution:</Typography>
                  <Typography fontWeight="bold">{processingInfo.max_resolution}</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2">Detection Threshold:</Typography>
                  <Typography fontWeight="bold">{processingInfo.threshold}</Typography>
                </Box>
                <Box display="flex" justifyContent="space-between">
                  <Typography variant="body2">Processing Time:</Typography>
                  <Typography fontWeight="bold" color="success.main">
                    {(latencyMs / 1000).toFixed(2)}s
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default EnhancedAnalyticsDashboard;
