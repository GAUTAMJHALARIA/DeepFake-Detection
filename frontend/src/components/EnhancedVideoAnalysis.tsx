import React, { useState, useCallback, useEffect } from 'react';
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
import { motion } from 'framer-motion';
import axios from 'axios';

import EnhancedVideoPlayer from './EnhancedVideoPlayer';
import ConfidenceHeatMap from './ConfidenceHeatMap';
import EnhancedAnalyticsDashboard from './EnhancedAnalyticsDashboard';
import GradCAMViewer from './GradCAMViewer';
import ErrorBoundary from './ErrorBoundary';
import ParticleBurst from './ui/ParticleBurst';

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

    const [showParticleBurst, setShowParticleBurst] = useState(false);

    useEffect(() => {
        if (uploading) {
            setShowParticleBurst(true);
        }
    }, [uploading]);

    return (
        <Box>
            {/* Particle Burst Animation */}
            <ParticleBurst
                trigger={showParticleBurst}
                onComplete={() => setShowParticleBurst(false)}
            />

            {!result && (
                <Box
                    sx={{
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        minHeight: '600px',
                        position: 'relative',
                    }}
                >
                    {/* Floating 3D Upload Capsule */}
                    <motion.div
                        initial={{ opacity: 0, y: 100, scale: 0.9 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        transition={{
                            duration: 0.8,
                            ease: [0.4, 0, 0.2, 1],
                            type: 'spring',
                            stiffness: 100,
                            damping: 20,
                        }}
                        style={{ width: '100%', display: 'flex', justifyContent: 'center' }}
                    >
                        <Box
                            {...getRootProps()}
                            sx={{
                                width: { xs: '90%', md: '650px' },
                                height: { xs: '450px', md: '550px' },
                                cursor: uploading ? 'not-allowed' : 'pointer',
                                position: 'relative',
                                perspective: '1200px',
                                perspectiveOrigin: 'center center',
                                filter: uploading ? 'none' : 'drop-shadow(0 20px 60px hsla(200, 100%, 55%, 0.3))',
                                transition: 'filter 0.4s ease',
                            }}
                        >
                        <input {...getInputProps()} />

                        {/* Floating Shadow */}
                        <Box
                            sx={{
                                position: 'absolute',
                                bottom: '-30px',
                                left: '50%',
                                transform: 'translateX(-50%)',
                                width: '60%',
                                height: '30px',
                                borderRadius: '50%',
                                background: uploading
                                    ? 'radial-gradient(ellipse, hsla(200, 100%, 55%, 0.4) 0%, transparent 70%)'
                                    : 'radial-gradient(ellipse, rgba(0,0,0,0.3) 0%, transparent 70%)',
                                filter: 'blur(20px)',
                                transition: 'all 0.4s ease',
                                animation: uploading ? 'shadowPulse 2s ease-in-out infinite' : 'none',
                            }}
                        />

                        {/* 3D Capsule Container */}
                        <Box
                            sx={{
                                width: '100%',
                                height: '100%',
                                position: 'relative',
                                transformStyle: 'preserve-3d',
                                transform: isDragActive
                                    ? 'rotateY(8deg) rotateX(-8deg) scale(1.08) translateY(-10px)'
                                    : uploading
                                    ? 'scale(0.92) translateY(5px)'
                                    : 'rotateY(0deg) rotateX(0deg)',
                                transition: 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
                                animation: uploading ? 'capsulePulse 1.5s ease-in-out infinite' : 'float 6s ease-in-out infinite',
                            }}
                        >
                            {/* Capsule Body - Enhanced Cylindrical Shape */}
                            <Box
                                sx={{
                                    position: 'absolute',
                                    top: '8%',
                                    left: '50%',
                                    transform: 'translateX(-50%)',
                                    width: '88%',
                                    height: '72%',
                                    borderRadius: '60px',
                                    background: uploading
                                        ? 'linear-gradient(135deg, hsla(200, 100%, 55%, 0.35) 0%, hsla(200, 100%, 55%, 0.25) 30%, hsla(300, 100%, 60%, 0.35) 60%, hsla(200, 100%, 55%, 0.35) 100%)'
                                        : isDragActive
                                        ? 'linear-gradient(135deg, hsla(200, 100%, 55%, 0.3) 0%, hsla(220, 35%, 12%, 0.98) 50%, hsla(200, 100%, 55%, 0.2) 100%)'
                                        : 'linear-gradient(135deg, hsla(220, 35%, 12%, 0.95) 0%, hsla(220, 30%, 15%, 0.85) 50%, hsla(220, 35%, 12%, 0.95) 100%)',
                                    backdropFilter: 'blur(40px) saturate(180%)',
                                    border: uploading
                                        ? '4px solid hsl(200, 100%, 55%)'
                                        : isDragActive
                                        ? '4px solid hsl(200, 100%, 55%)'
                                        : '2px solid hsla(200, 100%, 55%, 0.4)',
                                    boxShadow: uploading
                                        ? `
                                            0 0 80px hsla(200, 100%, 55%, 0.7),
                                            0 0 150px hsla(200, 100%, 55%, 0.5),
                                            0 0 200px hsla(300, 100%, 60%, 0.3),
                                            inset 0 0 80px hsla(200, 100%, 55%, 0.25),
                                            inset 0 -20px 60px hsla(200, 100%, 55%, 0.15),
                                            0 25px 80px rgba(0,0,0,0.6),
                                            0 50px 100px rgba(0,0,0,0.3)
                                        `
                                        : isDragActive
                                        ? `
                                            0 0 60px hsla(200, 100%, 55%, 0.6),
                                            inset 0 0 50px hsla(200, 100%, 55%, 0.2),
                                            inset 0 -15px 40px hsla(200, 100%, 55%, 0.1),
                                            0 20px 60px rgba(0,0,0,0.5),
                                            0 40px 80px rgba(0,0,0,0.3)
                                        `
                                        : `
                                            0 15px 50px rgba(0,0,0,0.4),
                                            inset 0 0 40px hsla(200, 100%, 55%, 0.08),
                                            inset 0 -10px 30px hsla(200, 100%, 55%, 0.05),
                                            0 30px 60px rgba(0,0,0,0.2)
                                        `,
                                    transition: 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
                                    overflow: 'hidden',
                                    '&::before': {
                                        content: '""',
                                        position: 'absolute',
                                        top: '-50%',
                                        left: '-50%',
                                        width: '200%',
                                        height: '200%',
                                        background: uploading
                                            ? 'conic-gradient(from 0deg, transparent 0%, hsl(200, 100%, 55%) 15%, hsl(200, 100%, 55%) 30%, hsl(300, 100%, 60%) 45%, transparent 60%)'
                                            : isDragActive
                                            ? 'linear-gradient(135deg, transparent, hsla(200, 100%, 55%, 0.15), transparent)'
                                            : 'linear-gradient(135deg, transparent, hsla(200, 100%, 55%, 0.05), transparent)',
                                        animation: uploading ? 'rotate 2s linear infinite' : isDragActive ? 'rotate 3s linear infinite' : 'none',
                                        opacity: uploading ? 0.8 : isDragActive ? 0.5 : 0.3,
                                    },
                                    '&::after': {
                                        content: '""',
                                        position: 'absolute',
                                        top: '20%',
                                        left: '10%',
                                        width: '80%',
                                        height: '60%',
                                        borderRadius: '50px',
                                        background: 'linear-gradient(180deg, hsla(200, 100%, 55%, 0.1) 0%, transparent 50%)',
                                        filter: 'blur(30px)',
                                        opacity: uploading ? 0.6 : 0.2,
                                        transition: 'opacity 0.4s ease',
                                    },
                                }}
                            >
                                {/* Pulsing Light Rings */}
                                {uploading && (
                                    <>
                                        <Box
                                            sx={{
                                                position: 'absolute',
                                                top: '50%',
                                                left: '50%',
                                                transform: 'translate(-50%, -50%)',
                                                width: '100%',
                                                height: '100%',
                                                borderRadius: '50px',
                                                border: '2px solid hsl(200, 100%, 55%)',
                                                animation: 'pulseRing 1.5s ease-out infinite',
                                                opacity: 0,
                                            }}
                                        />
                                        <Box
                                            sx={{
                                                position: 'absolute',
                                                top: '50%',
                                                left: '50%',
                                                transform: 'translate(-50%, -50%)',
                                                width: '100%',
                                                height: '100%',
                                                borderRadius: '50px',
                                                border: '2px solid hsl(200, 100%, 55%)',
                                                animation: 'pulseRing 1.5s ease-out infinite',
                                                animationDelay: '0.5s',
                                                opacity: 0,
                                            }}
                                        />
                                        <Box
                                            sx={{
                                                position: 'absolute',
                                                top: '50%',
                                                left: '50%',
                                                transform: 'translate(-50%, -50%)',
                                                width: '100%',
                                                height: '100%',
                                                borderRadius: '50px',
                                                border: '2px solid hsl(300, 100%, 60%)',
                                                animation: 'pulseRing 1.5s ease-out infinite',
                                                animationDelay: '1s',
                                                opacity: 0,
                                            }}
                                        />
                                    </>
                                )}

                                {/* Content */}
                                <Box
                                    sx={{
                                        position: 'relative',
                                        zIndex: 10,
                                        display: 'flex',
                                        flexDirection: 'column',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        height: '100%',
                                        p: 4,
                                        transform: uploading ? 'scale(0.9)' : 'scale(1)',
                                        transition: 'transform 0.4s ease',
                                    }}
                                >
                                    {/* Enhanced Icon Container with Pull-in Effect */}
                                    <Box
                                        sx={{
                                            width: '130px',
                                            height: '130px',
                                            borderRadius: '50%',
                                            background: uploading
                                                ? 'radial-gradient(circle, hsl(200, 100%, 55%) 0%, hsl(200, 100%, 55%) 30%, transparent 70%)'
                                                : isDragActive
                                                ? 'radial-gradient(circle, hsl(200, 100%, 55%) 0%, transparent 70%)'
                                                : 'radial-gradient(circle, hsla(200, 100%, 55%, 0.4) 0%, hsla(200, 100%, 55%, 0.2) 50%, transparent 70%)',
                                            display: 'flex',
                                            alignItems: 'center',
                                            justifyContent: 'center',
                                            mb: 3,
                                            transition: 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
                                            filter: uploading || isDragActive
                                                ? 'drop-shadow(0 0 50px hsl(200, 100%, 55%)) drop-shadow(0 0 80px hsl(200, 100%, 55%))'
                                                : 'drop-shadow(0 0 20px hsla(200, 100%, 55%, 0.3))',
                                            animation: uploading || isDragActive
                                                ? 'pullIn 0.8s cubic-bezier(0.4, 0, 0.2, 1), iconPulse 2s ease-in-out infinite'
                                                : 'iconFloat 4s ease-in-out infinite',
                                            transform: uploading
                                                ? 'scale(1.25) rotate(360deg)'
                                                : isDragActive
                                                ? 'scale(1.15)'
                                                : 'scale(1)',
                                            position: 'relative',
                                            '&::before': {
                                                content: '""',
                                                position: 'absolute',
                                                inset: '-10px',
                                                borderRadius: '50%',
                                                background: uploading
                                                    ? 'conic-gradient(from 0deg, hsl(200, 100%, 55%), hsl(200, 100%, 55%), hsl(300, 100%, 60%), hsl(200, 100%, 55%))'
                                                    : 'transparent',
                                                animation: uploading ? 'rotate 1.5s linear infinite' : 'none',
                                                opacity: uploading ? 0.6 : 0,
                                                filter: 'blur(8px)',
                                                transition: 'opacity 0.4s ease',
                                },
                            }}
                        >
                                        <CloudUpload
                                            sx={{
                                                fontSize: 56,
                                                color: uploading || isDragActive
                                                    ? 'hsl(200, 100%, 55%)'
                                                    : 'hsla(200, 100%, 55%, 0.8)',
                                                filter: uploading || isDragActive
                                                    ? 'drop-shadow(0 0 25px hsl(200, 100%, 55%)) drop-shadow(0 0 40px hsl(200, 100%, 55%))'
                                                    : 'drop-shadow(0 0 10px hsla(200, 100%, 55%, 0.4))',
                                                transform: uploading ? 'scale(0.75)' : 'scale(1)',
                                                transition: 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
                                                position: 'relative',
                                                zIndex: 2,
                                            }}
                                        />
                                    </Box>

                                    <Typography
                                        variant="h4"
                                        sx={{
                                            mb: 2,
                                            fontWeight: 900,
                                            fontSize: { xs: '1.8rem', md: '2.4rem' },
                                            letterSpacing: '-0.03em',
                                            textTransform: 'uppercase',
                                            background: uploading
                                                ? 'linear-gradient(135deg, hsl(200, 100%, 55%) 0%, hsl(200, 100%, 55%) 30%, hsl(300, 100%, 60%) 60%, hsl(200, 100%, 55%) 100%)'
                                                : isDragActive
                                                ? 'linear-gradient(135deg, hsl(200, 100%, 55%) 0%, hsl(200, 100%, 55%) 100%)'
                                                : 'linear-gradient(135deg, hsl(200, 100%, 55%) 0%, hsl(200, 100%, 55%) 100%)',
                                            backgroundSize: uploading ? '200% auto' : '100% auto',
                                            WebkitBackgroundClip: 'text',
                                            WebkitTextFillColor: 'transparent',
                                            backgroundClip: 'text',
                                            animation: uploading ? 'gradientText 2s ease infinite' : 'none',
                                            opacity: uploading ? 0.9 : 1,
                                            textShadow: uploading ? '0 0 30px hsla(200, 100%, 55%, 0.5)' : 'none',
                                            filter: uploading ? 'drop-shadow(0 0 20px hsl(200, 100%, 55%))' : 'none',
                                        }}
                                    >
                                        {uploading ? 'PROCESSING...' : isDragActive ? 'DROP HERE' : 'UPLOAD FILE'}
                            </Typography>

                                    <Typography
                                        variant="body2"
                                        color="text.secondary"
                                        sx={{
                                            mb: 3,
                                            textAlign: 'center',
                                            fontSize: '0.85rem',
                                            opacity: uploading ? 0.6 : 1,
                                        }}
                                    >
                                        {uploading
                                            ? 'File absorbed. Analyzing...'
                                            : isDragActive
                                            ? 'Release to upload'
                                            : 'Drag & drop or click to browse'}
                            </Typography>

                                    {/* File Format Chips - Animated */}
                                    {!uploading && (
                                        <Box
                                            sx={{
                                                display: 'flex',
                                                gap: 1,
                                                flexWrap: 'wrap',
                                                justifyContent: 'center',
                                                opacity: isDragActive ? 0.7 : 1,
                                            }}
                                        >
                                            {['MP4', 'AVI', 'MOV', 'MKV'].map((format, index) => (
                                                <Chip
                                                    key={format}
                                                    label={format}
                                                    size="small"
                                                    sx={{
                                                        background: 'hsla(200, 100%, 55%, 0.1)',
                                                        border: '1px solid hsla(200, 100%, 55%, 0.3)',
                                                        color: 'hsl(200, 100%, 55%)',
                                                        fontSize: '0.7rem',
                                                        animation: isDragActive
                                                            ? `fadeOut 0.3s ease ${index * 0.1}s forwards`
                                                            : 'none',
                                                    }}
                                                />
                                            ))}
                                        </Box>
                                    )}
                                </Box>
                            </Box>

                            {/* Enhanced Capsule Top Cap - 3D Effect */}
                            <Box
                                sx={{
                                    position: 'absolute',
                                    top: '3%',
                                    left: '50%',
                                    width: '72%',
                                    height: '18%',
                                    borderRadius: '60px 60px 25px 25px',
                                    background: uploading
                                        ? 'linear-gradient(180deg, hsla(200, 100%, 55%, 0.35) 0%, hsla(200, 100%, 55%, 0.2) 50%, transparent 100%)'
                                        : 'linear-gradient(180deg, hsla(200, 100%, 55%, 0.25) 0%, hsla(200, 100%, 55%, 0.1) 50%, transparent 100%)',
                                    border: uploading
                                        ? '3px solid hsl(200, 100%, 55%)'
                                        : '2px solid hsla(200, 100%, 55%, 0.4)',
                                    borderBottom: 'none',
                                    boxShadow: uploading
                                        ? '0 -15px 40px hsla(200, 100%, 55%, 0.4), inset 0 -5px 20px hsla(200, 100%, 55%, 0.2)'
                                        : '0 -8px 25px rgba(0,0,0,0.3), inset 0 -3px 15px hsla(200, 100%, 55%, 0.1)',
                                    transformStyle: 'preserve-3d',
                                    transform: 'translateX(-50%) rotateX(18deg) translateZ(5px)',
                                    transition: 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
                                    filter: uploading ? 'blur(0.5px)' : 'none',
                                }}
                            >
                                {/* Cap Highlight */}
                                <Box
                                    sx={{
                                        position: 'absolute',
                                        top: '30%',
                                        left: '20%',
                                        width: '60%',
                                        height: '30%',
                                        borderRadius: '50px',
                                        background: 'linear-gradient(90deg, transparent, hsla(200, 100%, 55%, 0.2), transparent)',
                                        filter: 'blur(10px)',
                                    }}
                                />
                            </Box>

                            {/* Enhanced Capsule Bottom Cap - 3D Effect */}
                            <Box
                                sx={{
                                    position: 'absolute',
                                    bottom: '3%',
                                    left: '50%',
                                    width: '72%',
                                    height: '18%',
                                    borderRadius: '25px 25px 60px 60px',
                                    background: uploading
                                        ? 'linear-gradient(0deg, hsla(200, 100%, 55%, 0.35) 0%, hsla(200, 100%, 55%, 0.2) 50%, transparent 100%)'
                                        : 'linear-gradient(0deg, hsla(200, 100%, 55%, 0.25) 0%, hsla(200, 100%, 55%, 0.1) 50%, transparent 100%)',
                                    border: uploading
                                        ? '3px solid hsl(200, 100%, 55%)'
                                        : '2px solid hsla(200, 100%, 55%, 0.4)',
                                    borderTop: 'none',
                                    boxShadow: uploading
                                        ? '0 15px 40px hsla(200, 100%, 55%, 0.4), inset 0 5px 20px hsla(200, 100%, 55%, 0.2)'
                                        : '0 8px 25px rgba(0,0,0,0.3), inset 0 3px 15px hsla(200, 100%, 55%, 0.1)',
                                    transformStyle: 'preserve-3d',
                                    transform: 'translateX(-50%) rotateX(-18deg) translateZ(5px)',
                                    transition: 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
                                    filter: uploading ? 'blur(0.5px)' : 'none',
                                }}
                            >
                                {/* Cap Highlight */}
                                <Box
                                    sx={{
                                        position: 'absolute',
                                        bottom: '30%',
                                        left: '20%',
                                        width: '60%',
                                        height: '30%',
                                        borderRadius: '50px',
                                        background: 'linear-gradient(90deg, transparent, hsla(200, 100%, 55%, 0.2), transparent)',
                                        filter: 'blur(10px)',
                                    }}
                                />
                            </Box>

                            {/* Enhanced Energy Particles - When Uploading */}
                            {uploading && (
                                <>
                                    {[...Array(20)].map((_, i) => (
                                        <Box
                                            key={i}
                                            sx={{
                                                position: 'absolute',
                                                top: '50%',
                                                left: '50%',
                                                width: i % 3 === 0 ? '6px' : i % 3 === 1 ? '4px' : '3px',
                                                height: i % 3 === 0 ? '6px' : i % 3 === 1 ? '4px' : '3px',
                                                borderRadius: '50%',
                                                background: `hsl(${180 + (i % 12) * 10}, 100%, ${50 + (i % 3) * 10}%)`,
                                                boxShadow: `0 0 15px hsl(${180 + (i % 12) * 10}, 100%, 50%), 0 0 30px hsl(${180 + (i % 12) * 10}, 100%, 50%)`,
                                                animation: `particleFloat 2.5s ease-in-out infinite`,
                                                animationDelay: `${i * 0.1}s`,
                                                transform: `translate(-50%, -50%) rotate(${i * 18}deg) translateY(-90px)`,
                                                opacity: 0.9,
                                            }}
                                        />
                                    ))}
                                    {/* Orbital Rings */}
                                    {[1, 2, 3].map((ring) => (
                                        <Box
                                            key={`ring-${ring}`}
                                            sx={{
                                                position: 'absolute',
                                                top: '50%',
                                                left: '50%',
                                                width: `${80 + ring * 40}px`,
                                                height: `${80 + ring * 40}px`,
                                                borderRadius: '50%',
                                                border: `1px solid hsla(${180 + ring * 20}, 100%, 50%, ${0.4 - ring * 0.1})`,
                                                transform: `translate(-50%, -50%) rotate(${ring * 45}deg)`,
                                                animation: `rotate ${2 + ring * 0.5}s linear infinite`,
                                                opacity: 0.6,
                                            }}
                                        />
                                    ))}
                                </>
                            )}
                        </Box>
                    </Box>
                    </motion.div>

                    {/* URL Upload Option - Smaller Side Panel */}
                    {!uploading && (
                        <Box
                            sx={{
                                position: 'absolute',
                                bottom: '20px',
                                right: '20px',
                                width: { xs: '90%', md: '300px' },
                            }}
                        >
                            <Paper
                                sx={{
                                    p: 2,
                                    borderRadius: '16px',
                                    background: uploadMode === 'url'
                                        ? 'linear-gradient(135deg, hsla(300, 100%, 60%, 0.15) 0%, hsla(220, 35%, 12%, 0.9) 100%)'
                                        : 'linear-gradient(135deg, hsla(220, 35%, 12%, 0.8) 0%, hsla(220, 30%, 15%, 0.6) 100%)',
                                    backdropFilter: 'blur(20px)',
                                    border: uploadMode === 'url'
                                        ? '2px solid hsl(300, 100%, 60%)'
                                        : '2px solid hsla(300, 100%, 60%, 0.2)',
                                    cursor: 'pointer',
                                    transition: 'all 0.3s ease',
                                    '&:hover': {
                                        borderColor: 'hsl(300, 100%, 60%)',
                                        boxShadow: '0 0 30px hsla(300, 100%, 60%, 0.4)',
                                    },
                                }}
                                onClick={() => setUploadMode('url')}
                            >
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2 }}>
                                    <LinkIcon sx={{ color: 'hsl(300, 100%, 60%)', fontSize: 24 }} />
                                    <Typography
                                        variant="h6"
                                        sx={{
                                            fontWeight: 700,
                                            fontSize: '0.9rem',
                                            background: 'linear-gradient(135deg, hsl(300, 100%, 60%) 0%, hsl(200, 100%, 55%) 100%)',
                                            WebkitBackgroundClip: 'text',
                                            WebkitTextFillColor: 'transparent',
                                        }}
                                    >
                                        From URL
                                    </Typography>
                                </Box>
                    {uploadMode === 'url' && (
                                    <Box sx={{ display: 'flex', gap: 1 }}>
                                <TextField
                                    fullWidth
                                            size="small"
                                            placeholder="Paste URL..."
                                    value={urlInput}
                                    onChange={(e) => setUrlInput(e.target.value)}
                                            onClick={(e) => e.stopPropagation()}
                                            onFocus={() => setUploadMode('url')}
                                            sx={{
                                                '& .MuiOutlinedInput-root': {
                                                    background: 'hsla(220, 35%, 12%, 0.6)',
                                                    border: '1px solid hsla(300, 100%, 60%, 0.3)',
                                                },
                                            }}
                                />
                                <Button
                                            size="small"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                handleUrlUpload();
                                            }}
                                            disabled={!urlInput.trim()}
                                            sx={{
                                                background: 'linear-gradient(135deg, hsl(300, 100%, 60%) 0%, hsl(200, 100%, 55%) 100%)',
                                                color: 'hsl(220, 40%, 8%)',
                                                minWidth: '80px',
                                                fontWeight: 700,
                                            }}
                                >
                                    Analyze
                                </Button>
                            </Box>
                                )}
                        </Paper>
                        </Box>
                    )}
                </Box>
            )}

            {uploading && (
                <Card
                    sx={{
                        mb: 3,
                        background: 'linear-gradient(135deg, hsla(220, 35%, 12%, 0.9) 0%, hsla(220, 30%, 15%, 0.8) 100%)',
                        border: '1px solid hsla(200, 100%, 55%, 0.3)',
                        boxShadow: '0 0 20px hsla(200, 100%, 55%, 0.2)',
                    }}
                >
                    <CardContent>
                        <Typography
                            variant="h6"
                            gutterBottom
                            sx={{
                                color: 'hsl(200, 100%, 55%)',
                                fontWeight: 600,
                                letterSpacing: '0.02em',
                                textTransform: 'uppercase',
                            }}
                        >
                            Processing Video...
                        </Typography>
                        <LinearProgress
                            variant="determinate"
                            value={progress}
                            sx={{
                                mb: 2,
                                height: 10,
                                borderRadius: '5px',
                                backgroundColor: 'hsla(200, 100%, 55%, 0.1)',
                                '& .MuiLinearProgress-bar': {
                                    background: 'linear-gradient(90deg, hsl(200, 100%, 55%) 0%, hsl(200, 100%, 55%) 100%)',
                                    boxShadow: '0 0 10px hsl(200, 100%, 55%)',
                                },
                            }}
                        />
                        <Typography
                            variant="body2"
                            color="textSecondary"
                            sx={{
                                letterSpacing: '0.01em',
                            }}
                        >
                            {processingStep}
                        </Typography>
                        <Typography
                            variant="caption"
                            color="textSecondary"
                            display="block"
                            sx={{
                                mt: 1,
                                letterSpacing: '0.01em',
                                opacity: 0.8,
                            }}
                        >
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
                    <Card
                        sx={{
                            mb: 3,
                            background: 'linear-gradient(135deg, hsla(220, 35%, 12%, 0.9) 0%, hsla(220, 30%, 15%, 0.8) 100%)',
                            border: '1px solid hsla(200, 100%, 55%, 0.3)',
                            boxShadow: '0 0 20px hsla(200, 100%, 55%, 0.2)',
                        }}
                    >
                        <CardContent>
                            <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                                <Typography
                                    variant="h5"
                                    sx={{
                                        background: 'linear-gradient(135deg, hsl(200, 100%, 55%) 0%, hsl(200, 100%, 55%) 100%)',
                                        WebkitBackgroundClip: 'text',
                                        WebkitTextFillColor: 'transparent',
                                        backgroundClip: 'text',
                                        fontWeight: 700,
                                        letterSpacing: '0.02em',
                                        textTransform: 'uppercase',
                                    }}
                                >
                                    Analysis Results
                                </Typography>
                                <Chip
                                    icon={result.label === 'fake' ? <VideoFile /> : <VideoFile />}
                                    label={`${result.label.toUpperCase()} - ${(result.score * 100).toFixed(1)}%`}
                                    sx={{
                                        fontSize: '0.9rem',
                                        padding: '8px 16px',
                                        fontWeight: 600,
                                        letterSpacing: '0.05em',
                                        background: result.label === 'fake'
                                            ? 'hsla(0, 100%, 65%, 0.2)'
                                            : 'hsla(150, 70%, 50%, 0.2)',
                                        border: `1px solid ${result.label === 'fake' ? 'hsl(0, 100%, 65%)' : 'hsl(150, 70%, 50%)'}`,
                                        color: result.label === 'fake' ? 'hsl(0, 100%, 65%)' : 'hsl(150, 70%, 50%)',
                                        boxShadow: `0 0 15px ${result.label === 'fake' ? 'hsla(0, 100%, 65%, 0.3)' : 'hsla(150, 70%, 50%, 0.3)'}`,
                                    }}
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
