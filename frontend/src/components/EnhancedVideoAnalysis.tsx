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
    Grid,
    Chip,
    Divider,
    TextField,
    Tabs,
    Tab,
} from '@mui/material';
import {
    CloudUpload,
    VideoFile,
    Analytics,
    Timeline,
    Visibility,
    Link as LinkIcon,
    InsertDriveFile,
} from '@mui/icons-material';
import axios from 'axios';

import EnhancedVideoPlayer from './EnhancedVideoPlayer';
import ConfidenceHeatMap from './ConfidenceHeatMap';
import AnalyticsDashboard from './AnalyticsDashboard';
import EnhancedAnalyticsDashboard from './EnhancedAnalyticsDashboard';
import GradCAMViewer from './GradCAMViewer';
import ErrorBoundary from './ErrorBoundary';

interface EnhancedAnalysisResult {
    id: string;
    score: number;
    label: string;
    video_info: {
        duration: number;
        fps: number;
        total_frames: number;
        processed_frames: number;
        resolution: string;
        face_detect_rate: number;
    };
    frames: Array<{
        index: number;
        timestamp: number;
        confidence: number;
        label: string;
        face_detected: boolean;
        confidence_color: [number, number, number];
        has_gradcam: boolean;
    }>;
    statistics: {
        mean_confidence: number;
        confidence_variance: number;
        max_confidence: number;
        min_confidence: number;
        suspicious_frames: number;
        total_frames: number;
        quality_score: number;
    };
    processing_info: {
        gradcam_enabled: boolean;
        all_frames_extracted: boolean;
        max_resolution: string;
        threshold: number;
    };
    latency_ms: number;
    version: string;
}

const EnhancedVideoAnalysis: React.FC = () => {
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [result, setResult] = useState<EnhancedAnalysisResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [uploadedFile, setUploadedFile] = useState<File | null>(null);
    const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
    const [processingStep, setProcessingStep] = useState('');
    const [urlInput, setUrlInput] = useState('');
    const [uploadMode, setUploadMode] = useState<'file' | 'url'>('file');

    const formatErrorMessage = (error: any): string => {
        if (typeof error === 'string') {
            return error;
        }
        if (error?.response?.data?.detail) {
            const detail = error.response.data.detail;
            if (typeof detail === 'string') {
                return detail;
            }
            if (Array.isArray(detail)) {
                return detail.map((d: any) => d.msg || JSON.stringify(d)).join(', ');
            }
        }
        return 'Analysis failed. Please try again.';
    };

    const onDrop = useCallback(async (acceptedFiles: File[]) => {
        const file = acceptedFiles[0];
        if (!file) return;

        console.log('File received:', file.name, file.size, file.type);

        // Read the file as ArrayBuffer to create a proper copy
        try {
            const arrayBuffer = await file.arrayBuffer();
            const fileCopy = new File([arrayBuffer], file.name, { type: file.type });
            console.log('File copy created:', fileCopy.name, fileCopy.size, fileCopy.type);
            setUploadedFile(fileCopy);
        } catch (error) {
            console.error('Failed to create file copy:', error);
            setUploadedFile(file); // Fallback to original file
        }

        setUploading(true);
        setProgress(0);
        setError(null);
        setResult(null);
        setProcessingStep('Uploading file...');

        try {
            const formData = new FormData();
            formData.append('file', file); // Use original file for upload

            const response = await axios.post('http://localhost:8000/predict', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                    'Authorization': 'Bearer change-me',
                },
                onUploadProgress: (progressEvent) => {
                    if (progressEvent.total) {
                        const percentCompleted = Math.round(
                            (progressEvent.loaded * 100) / progressEvent.total
                        );
                        setProgress(Math.min(percentCompleted, 90)); // Reserve 10% for processing

                        if (percentCompleted < 100) {
                            setProcessingStep('Uploading file...');
                        } else {
                            setProcessingStep('Processing video...');
                        }
                    }
                },
            });

            setProgress(100);
            setProcessingStep('Analysis complete!');
            setResult(response.data);

        } catch (err: any) {
            setError(formatErrorMessage(err));
            setProcessingStep('Analysis failed');
        } finally {
            setUploading(false);
        }
    }, []);

    const handleUrlUpload = useCallback(async () => {
        if (!urlInput.trim()) {
            setError('Please enter a valid URL');
            return;
        }

        setUploading(true);
        setProgress(0);
        setError(null);
        setResult(null);
        setUploadedFile(null); // No file for URL upload

        try {
            setProcessingStep('Downloading video from URL...');
            setProgress(20);

            const response = await axios.post(
                'http://localhost:8000/predict-url',
                { url: urlInput.trim() },
                { headers: { 'Content-Type': 'application/json' } }
            );

            setProgress(100);
            setProcessingStep('Analysis complete!');
            setResult(response.data);

        } catch (err: any) {
            setError(formatErrorMessage(err));
            setProcessingStep('Analysis failed');
        } finally {
            setUploading(false);
        }
    }, [urlInput]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
        },
        multiple: false,
        disabled: uploading,
    });

    const handleFrameChange = (frameIndex: number) => {
        setCurrentFrameIndex(frameIndex);
    };

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
            <Typography variant="h4" gutterBottom>
                Deepfake Detection Analysis
            </Typography>
            <Typography variant="body1" color="textSecondary" paragraph>
                Advanced AI-powered deepfake detection with frame-by-frame analysis, confidence visualization, and explainable AI features.
            </Typography>

            {!result && (
                <Box>
                    {/* Upload Mode Tabs */}
                    <Paper sx={{ mb: 2 }}>
                        <Tabs
                            value={uploadMode}
                            onChange={(_, newValue) => setUploadMode(newValue)}
                            centered
                        >
                            <Tab
                                icon={<InsertDriveFile />}
                                label="Upload File"
                                value="file"
                                disabled={uploading}
                            />
                            <Tab
                                icon={<LinkIcon />}
                                label="From URL"
                                value="url"
                                disabled={uploading}
                            />
                        </Tabs>
                    </Paper>

                    {/* File Upload Mode */}
                    {uploadMode === 'file' && (
                        <Paper
                            {...getRootProps()}
                            sx={{
                                p: 6,
                                textAlign: 'center',
                                cursor: uploading ? 'not-allowed' : 'pointer',
                                border: '2px dashed',
                                borderColor: isDragActive ? 'primary.main' : 'grey.500',
                                backgroundColor: isDragActive ? 'action.hover' : 'background.paper',
                                transition: 'all 0.3s ease',
                                '&:hover': {
                                    borderColor: 'primary.main',
                                },
                            }}
                        >
                            <input {...getInputProps()} />
                            <CloudUpload sx={{ fontSize: 64, color: 'grey.400', mb: 2 }} />
                            <Typography variant="h6" gutterBottom>
                                {isDragActive ? 'Drop your video here' : 'Drag & drop or click to upload a video'}
                            </Typography>
                            <Typography variant="body1" color="textSecondary" sx={{ mb: 3 }}>
                                Supported formats: MP4, AVI, MOV, MKV, WebM (up to 1080p)
                            </Typography>
                            <Button variant="contained" size="large" disabled={uploading}>
                                Select Video File
                            </Button>
                        </Paper>
                    )}

                    {/* URL Upload Mode */}
                    {uploadMode === 'url' && (
                        <Paper sx={{ p: 4 }}>
                            <Box sx={{ display: 'flex', gap: 2 }}>
                                <TextField
                                    fullWidth
                                    label="Enter Video URL"
                                    placeholder="https://www.youtube.com/watch?v=..."
                                    value={urlInput}
                                    onChange={(e) => setUrlInput(e.target.value)}
                                    disabled={uploading}
                                    helperText="Supports YouTube, Twitter, Instagram, TikTok, Vimeo, and more"
                                />
                                <Button
                                    variant="contained"
                                    size="large"
                                    onClick={handleUrlUpload}
                                    disabled={uploading || !urlInput.trim()}
                                    startIcon={<LinkIcon />}
                                    sx={{ minWidth: 150 }}
                                >
                                    Analyze
                                </Button>
                            </Box>
                        </Paper>
                    )}
                </Box>
            )}

            {uploading && (
                <Card sx={{ mb: 3 }}>
                    <CardContent>
                        <Typography variant="h6" gutterBottom>
                            Processing Video...
                        </Typography>
                        <LinearProgress variant="determinate" value={progress} sx={{ mb: 2, height: 8 }} />
                        <Typography variant="body2" color="textSecondary">
                            {processingStep}
                        </Typography>
                        <Typography variant="caption" color="textSecondary" display="block" sx={{ mt: 1 }}>
                            This may take a few minutes for longer videos. We're extracting all frames and generating analysis data.
                        </Typography>
                    </CardContent>
                </Card>
            )}

            {error && (
                <Alert severity="error" sx={{ mb: 3 }}>
                    <Typography variant="body2">{error}</Typography>
                </Alert>
            )}

            {result && (
                <Box>
                    {/* Overall Results Summary */}
                    <Card sx={{ mb: 3 }}>
                        <CardContent>
                            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                                <Typography variant="h5">Analysis Results</Typography>
                                <Chip
                                    icon={result.label === 'fake' ? <VideoFile /> : <VideoFile />}
                                    label={`${result.label.toUpperCase()} - ${(result.score * 100).toFixed(1)}%`}
                                    color={result.label === 'fake' ? 'error' : 'success'}
                                    variant="filled"
                                    sx={{ fontSize: '1rem', padding: '8px 16px' }}
                                />
                            </Box>

                            <Box mb={3}>
                                <Typography variant="body2" color="textSecondary" gutterBottom>
                                    Overall Deepfake Confidence
                                </Typography>
                                <Box display="flex" alignItems="center" gap={2}>
                                    <LinearProgress
                                        variant="determinate"
                                        value={result.score * 100}
                                        color={getScoreColor(result.score)}
                                        sx={{ flexGrow: 1, height: 12, borderRadius: 6 }}
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

                            <Grid container spacing={2}>
                                <Grid item xs={6} md={3}>
                                    <Typography variant="body2" color="textSecondary">
                                        Duration
                                    </Typography>
                                    <Typography variant="h6">
                                        {result.video_info.duration.toFixed(1)}s
                                    </Typography>
                                </Grid>
                                <Grid item xs={6} md={3}>
                                    <Typography variant="body2" color="textSecondary">
                                        Frames Analyzed
                                    </Typography>
                                    <Typography variant="h6">
                                        {result.video_info.processed_frames}
                                    </Typography>
                                </Grid>
                                <Grid item xs={6} md={3}>
                                    <Typography variant="body2" color="textSecondary">
                                        Face Detection Rate
                                    </Typography>
                                    <Typography variant="h6">
                                        {(result.video_info.face_detect_rate * 100).toFixed(1)}%
                                    </Typography>
                                </Grid>
                                <Grid item xs={6} md={3}>
                                    <Typography variant="body2" color="textSecondary">
                                        Processing Time
                                    </Typography>
                                    <Typography variant="h6">
                                        {(result.latency_ms / 1000).toFixed(1)}s
                                    </Typography>
                                </Grid>
                            </Grid>
                        </CardContent>
                    </Card>

                    {/* Video Player */}
                    <ErrorBoundary>
                        <EnhancedVideoPlayer
                            videoFile={uploadedFile}
                            analysisData={result}
                            onFrameChange={handleFrameChange}
                        />
                    </ErrorBoundary>

                    {/* Additional Analysis Components */}
                    <Box sx={{ mt: 3 }}>
                        <Grid container spacing={3}>
                            <Grid item xs={12} md={6}>
                                <ConfidenceHeatMap
                                    analysisData={result}
                                    currentFrameIndex={currentFrameIndex}
                                    onFrameSelect={handleFrameChange}
                                />
                            </Grid>
                            <Grid item xs={12} md={6}>
                                <EnhancedAnalyticsDashboard
                                    statistics={result.statistics}
                                    videoInfo={result.video_info}
                                    processingInfo={result.processing_info}
                                    latencyMs={result.latency_ms}
                                />
                            </Grid>
                            <Grid item xs={12}>
                                <GradCAMViewer
                                    analysisId={result.id}
                                    currentFrameIndex={currentFrameIndex}
                                    totalFrames={result.frames.length}
                                    onFrameChange={handleFrameChange}
                                />
                            </Grid>
                        </Grid>
                    </Box>
                </Box>
            )}
        </Box>
    );
};

export default EnhancedVideoAnalysis;
