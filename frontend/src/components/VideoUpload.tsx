import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Box,
  Paper,
  Typography,
  Button,
  LinearProgress,
  Alert,
  Card,
  CardContent,
  Chip,
  Divider,
  Slider,
} from '@mui/material';
import {
  CloudUpload,
  CheckCircle,
  Error,
} from '@mui/icons-material';
import axios from 'axios';

interface AnalysisResult {
  id: string;
  score: number;
  label: string;
  frame_samples: Array<{ t: number; score: number }>;
  version: string;
  latency_ms: number;
  meta: any;
}

const VideoUpload: React.FC = () => {
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(2.0);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setUploading(true);
    setProgress(0);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const isImage = file.type.startsWith('image/');
      const endpoint = isImage ? '/predict-image' : '/predict';
      const url = `http://localhost:8000${endpoint}${!isImage ? `?fps=${fps}` : ''}`;

      const response = await axios.post(url, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'Authorization': 'Bearer change-me',
        },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setProgress(percentCompleted);
          }
        },
      });

      setResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Analysis failed');
    } finally {
      setUploading(false);
    }
  }, [fps]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
      'image/*': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'],
    },
    multiple: false,
    disabled: uploading,
  });

  const getScoreColor = (score: number) => {
    if (score >= 0.7) return 'error';
    if (score >= 0.3) return 'warning';
    return 'success';
  };

  const getScoreLabel = (score: number) => {
    if (score >= 0.8) return 'High Confidence Fake';
    if (score >= 0.6) return 'Likely Fake';
    if (score >= 0.4) return 'Uncertain';
    if (score >= 0.2) return 'Likely Real';
    return 'High Confidence Real';
  };

  return (
    <Box>
      <Box display="flex" gap={3} flexDirection={{ xs: 'column', md: 'row' }}>
        {/* Upload Section */}
        <Box flex={1}>
          <Paper
            {...getRootProps()}
            sx={{
              p: 4,
              textAlign: 'center',
              cursor: uploading ? 'not-allowed' : 'pointer',
              border: '2px dashed',
              borderColor: isDragActive ? 'primary.main' : 'grey.500',
              backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
              transition: 'all 0.3s ease',
              '&:hover': {
                borderColor: 'primary.main',
                backgroundColor: 'action.hover',
              },
            }}
          >
            <input {...getInputProps()} />
            <CloudUpload sx={{ fontSize: 64, color: 'primary.main', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              {isDragActive
                ? 'Drop your video or image here'
                : 'Drag & drop a video/image, or click to select'}
            </Typography>
            <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
              Supported formats: MP4, AVI, MOV, MKV, WebM, JPG, PNG, etc.
            </Typography>
            <Button variant="contained" disabled={uploading}>
              Select File
            </Button>
          </Paper>

          {/* FPS Control for Videos */}
          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Processing Settings
              </Typography>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Frame sampling rate (FPS): {fps}
              </Typography>
              <Slider
                value={fps}
                onChange={(_, value) => setFps(value as number)}
                min={0.5}
                max={10}
                step={0.5}
                marks={[
                  { value: 1, label: '1' },
                  { value: 2, label: '2' },
                  { value: 5, label: '5' },
                  { value: 10, label: '10' },
                ]}
                disabled={uploading}
              />
              <Typography variant="caption" color="textSecondary">
                Higher FPS = more frames analyzed = better accuracy but slower processing
              </Typography>
            </CardContent>
          </Card>
        </Box>

        {/* Results Section */}
        <Box flex={1}>
          {uploading && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Processing...
                </Typography>
                <LinearProgress variant="determinate" value={progress} sx={{ mb: 2 }} />
                <Typography variant="body2" color="textSecondary">
                  Analyzing your file...
                </Typography>
              </CardContent>
            </Card>
          )}

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              <Typography variant="body2">{error}</Typography>
            </Alert>
          )}

          {result && (
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
                  <Typography variant="h6">Analysis Results</Typography>
                  <Chip
                    icon={result.label === 'fake' ? <Error /> : <CheckCircle />}
                    label={result.label.toUpperCase()}
                    color={result.label === 'fake' ? 'error' : 'success'}
                    variant="filled"
                  />
                </Box>

                <Box mb={3}>
                  <Typography variant="body2" color="textSecondary" gutterBottom>
                    Deepfake Confidence Score
                  </Typography>
                  <Box display="flex" alignItems="center" gap={2}>
                    <LinearProgress
                      variant="determinate"
                      value={result.score * 100}
                      color={getScoreColor(result.score)}
                      sx={{ flexGrow: 1, height: 8, borderRadius: 4 }}
                    />
                    <Typography variant="h6" color={`${getScoreColor(result.score)}.main`}>
                      {(result.score * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                  <Typography variant="caption" color="textSecondary">
                    {getScoreLabel(result.score)}
                  </Typography>
                </Box>

                <Divider sx={{ my: 2 }} />

                <Box display="flex" flexWrap="wrap" gap={2}>
                  <Box>
                    <Typography variant="body2" color="textSecondary">
                      Processing Time
                    </Typography>
                    <Typography variant="body1">
                      {result.latency_ms}ms
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="body2" color="textSecondary">
                      Frames Analyzed
                    </Typography>
                    <Typography variant="body1">
                      {result.frame_samples?.length || 1}
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="body2" color="textSecondary">
                      Model Version
                    </Typography>
                    <Typography variant="body1">
                      v{result.version}
                    </Typography>
                  </Box>
                  <Box>
                    <Typography variant="body2" color="textSecondary">
                      Analysis ID
                    </Typography>
                    <Typography variant="body2" sx={{ fontSize: '0.8rem' }}>
                      {result.id.substring(0, 8)}...
                    </Typography>
                  </Box>
                </Box>

                {result.frame_samples && result.frame_samples.length > 1 && (
                  <Box mt={3}>
                    <Typography variant="subtitle2" gutterBottom>
                      Frame Analysis Summary
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Highest confidence: {Math.max(...result.frame_samples.map(f => f.score * 100)).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Lowest confidence: {Math.min(...result.frame_samples.map(f => f.score * 100)).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2" color="textSecondary">
                      Average confidence: {(result.frame_samples.reduce((sum, f) => sum + f.score, 0) / result.frame_samples.length * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                )}
              </CardContent>
            </Card>
          )}
        </Box>
      </Box>
    </Box>
  );
};

export default VideoUpload;
