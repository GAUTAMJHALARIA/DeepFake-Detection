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
    Tabs,
    Tab,
    Grid,
    Chip,
    Divider,
} from '@mui/material';
import {
    CloudUpload,
    VideoFile,
    Analytics,
    Timeline,
    Visibility,
} from '@mui/icons-material';
import axios from 'axios';

import EnhancedVideoPlayer from './EnhancedVideoPlayer';
import ConfidenceHeatMap from './ConfidenceHeatMap';
import AnalyticsDashboard from './AnalyticsDashboard';
import GradCAMViewer from './GradCAMViewer';

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

interface TabPanelProps {
    children?: React.ReactNode;
    index: number;
    value: number;
}

function TabPanel(props: TabPanelProps) {
    const { children, value, index, ...other } = props;
    return (
        <div
            role="tabpanel"
            hidden={value !== index}
            id={`tabpanel-${index}`}
            {...other}
        >
            {value === index && <Box>{children}</Box>}
        </div>
    );
}

const EnhancedVideoAnalysis: React.FC = () => {
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [result, setResult] = useState<EnhancedAnalysisResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [uploadedFile, setUploadedFile] = useState<File | null>(null);
    const [tabValue, setTabValue] = useState(0);
    const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
    const [processingStep, setProcessingStep] = useState('');

    const onDrop = useCallback(async (acceptedFiles: File[]) => {
        const file = acceptedFiles[0];
        if (!file) return;

        setUploadedFile(file);
        setUploading(true);
        setProgress(0);
        setError(null);
        setResult(null);
        setProcessingStep('Uploading file...');

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await axios.post('http://localhost:8000/predict-enhanced', formData, {
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
            setError(err.response?.data?.detail || 'Enhanced analysis failed');
            setProcessingStep('Analysis failed');
        } finally {
            setUploading(false);
        }
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
        },
        multiple: false,
        disabled: uploading,
    });

    const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
        setTabValue(newValue);
    };

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
                Enhanced Video Analysis
            </Typography>
            <Typography variant="body1" color="textSecondary" paragraph>
                Advanced deepfake detection with frame-by-frame analysis, confidence visualization, and explainable AI features.
            </Typography>

            {!result && (
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
                            backgroundColor: 'action.hover',
                        },
                        mb: 3,
                    }}
                >
                    <input {...getInputProps()} />
                    <CloudUpload sx={{ fontSize: 80, color: 'primary.main', mb: 2 }} />
                    <Typography variant="h5" gutterBottom>
                        {isDragActive
                            ? 'Drop your video here'
                            : 'Drag & drop a video, or click to select'}
                    </Typography>
                    <Typography variant="body1" color="textSecondary" sx={{ mb: 3 }}>
                        Supported formats: MP4, AVI, MOV, MKV, WebM (up to 1080p)
                    </Typography>
                    <Button variant="contained" size="large" disabled={uploading}>
                        Select Video File
                    </Button>
                </Paper>
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

            {result && uploadedFile && (
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

                    {/* Tabbed Interface */}
                    <Paper sx={{ mb: 3 }}>
                        <Tabs value={tabValue} onChange={handleTabChange} variant="scrollable" scrollButtons="auto">
                            <Tab icon={<VideoFile />} label="Video Player" />
                            <Tab icon={<Timeline />} label="Heat Map" />
                            <Tab icon={<Analytics />} label="Analytics" />
                            <Tab icon={<Visibility />} label="Explainability" />
                        </Tabs>
                    </Paper>

                    {/* Tab Panels */}
                    <TabPanel value={tabValue} index={0}>
                        <EnhancedVideoPlayer
                            videoFile={uploadedFile}
                            analysisData={result}
                            onFrameChange={handleFrameChange}
                        />
                    </TabPanel>

                    <TabPanel value={tabValue} index={1}>
                        <ConfidenceHeatMap
                            analysisData={result}
                            currentFrameIndex={currentFrameIndex}
                            onFrameSelect={handleFrameChange}
                        />
                    </TabPanel>

                    <TabPanel value={tabValue} index={2}>
                        <AnalyticsDashboard
                            statistics={result.statistics}
                            videoInfo={result.video_info}
                            processingInfo={result.processing_info}
                            latencyMs={result.latency_ms}
                        />
                    </TabPanel>

                    <TabPanel value={tabValue} index={3}>
                        <GradCAMViewer
                            analysisId={result.id}
                            currentFrameIndex={currentFrameIndex}
                            totalFrames={result.frames.length}
                            onFrameChange={handleFrameChange}
                        />
                    </TabPanel>
                </Box>
            )}
        </Box>
    );
};

export default EnhancedVideoAnalysis;
