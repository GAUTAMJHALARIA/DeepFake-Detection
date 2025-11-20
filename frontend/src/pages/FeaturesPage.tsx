import React from 'react';
import { Box, Container, Typography, Grid, Card, CardContent } from '@mui/material';
import {
  CloudUpload,
  Link as LinkIcon,
  Timeline,
  BarChart,
  Psychology,
  Speed,
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import CreativeFeatures from '../components/CreativeFeatures';

const detailedFeatures = [
  {
    icon: <CloudUpload sx={{ fontSize: 50 }} />,
    title: 'Drag & Drop Upload',
    description: 'Intuitive file upload with drag-and-drop interface. Supports MP4, AVI, MOV, MKV, WebM formats up to 1080p resolution.',
    color: 'hsl(200, 100%, 55%)',
  },
  {
    icon: <LinkIcon sx={{ fontSize: 50 }} />,
    title: 'URL Support',
    description: 'Analyze videos directly from YouTube, Twitter, Instagram, TikTok, Vimeo without downloading. Automatic processing.',
    color: 'hsl(200, 100%, 55%)',
  },
  {
    icon: <Timeline sx={{ fontSize: 50 }} />,
    title: 'Frame-by-Frame Analysis',
    description: 'Comprehensive analysis of every frame with individual confidence scores, face detection, and timestamp tracking.',
    color: 'hsl(300, 100%, 60%)',
  },
  {
    icon: <BarChart sx={{ fontSize: 50 }} />,
    title: 'Interactive Heat Maps',
    description: 'Visualize confidence levels across the entire video timeline with clickable heat maps and frame thumbnails.',
    color: 'hsl(200, 100%, 55%)',
  },
  {
    icon: <Psychology sx={{ fontSize: 50 }} />,
    title: 'Grad-CAM++ Visualization',
    description: 'Explainable AI with Grad-CAM++ heatmaps showing exactly which regions influence the model\'s decision.',
    color: 'hsl(200, 100%, 55%)',
  },
  {
    icon: <Speed sx={{ fontSize: 50 }} />,
    title: 'Real-Time Processing',
    description: 'Fast analysis with real-time progress tracking. Optimized algorithms for quick results without compromising accuracy.',
    color: 'hsl(300, 100%, 60%)',
  },
];

const FeaturesPage: React.FC = () => {
  return (
    <Box sx={{ minHeight: 'calc(100vh - 64px)', py: { xs: 6, md: 10 } }}>
      <Container maxWidth="xl">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <Box sx={{ textAlign: 'center', mb: 8 }}>
            <Typography
              variant="overline"
              sx={{
                display: 'block',
                mb: 2,
                color: 'hsl(200, 100%, 55%)',
                fontSize: '1rem',
                letterSpacing: '0.3em',
                fontWeight: 600,
              }}
            >
              COMPREHENSIVE FEATURES
            </Typography>
            <Typography
              variant="h2"
              sx={{
                fontSize: { xs: '2.5rem', md: '4rem' },
                fontWeight: 900,
                mb: 3,
                background: 'linear-gradient(135deg, hsl(200, 100%, 55%) 0%, hsl(300, 100%, 60%) 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                backgroundClip: 'text',
                letterSpacing: '-0.03em',
                textTransform: 'uppercase',
              }}
            >
              Advanced Capabilities
            </Typography>
            <Typography
              variant="h6"
              color="text.secondary"
              sx={{
                maxWidth: '700px',
                mx: 'auto',
                fontWeight: 300,
                lineHeight: 1.7,
              }}
            >
              Everything you need for comprehensive deepfake detection and analysis
              with cutting-edge AI technology
            </Typography>
          </Box>
        </motion.div>

        <Grid container spacing={4}>
          {detailedFeatures.map((feature, index) => (
            <Grid item xs={12} md={6} lg={4} key={index}>
              <motion.div
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                whileHover={{ y: -5 }}
              >
                <Card
                  sx={{
                    height: '100%',
                    background: 'linear-gradient(135deg, hsla(220, 35%, 12%, 0.9) 0%, hsla(220, 30%, 15%, 0.8) 100%)',
                    backdropFilter: 'blur(20px)',
                    border: `1px solid ${feature.color}33`,
                    borderRadius: '20px',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      borderColor: feature.color,
                      boxShadow: `0 0 40px ${feature.color}44, 0 10px 30px rgba(0,0,0,0.3)`,
                      transform: 'translateY(-5px)',
                    },
                  }}
                >
                  <CardContent sx={{ p: 4 }}>
                    <Box
                      sx={{
                        color: feature.color,
                        mb: 2,
                        filter: `drop-shadow(0 0 15px ${feature.color}88)`,
                      }}
                    >
                      {feature.icon}
                    </Box>
                    <Typography
                      variant="h5"
                      sx={{
                        mb: 2,
                        fontWeight: 700,
                        fontSize: { xs: '1.3rem', md: '1.5rem' },
                        letterSpacing: '0.02em',
                        textTransform: 'uppercase',
                      }}
                    >
                      {feature.title}
                    </Typography>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{
                        lineHeight: 1.8,
                        fontSize: '0.95rem',
                      }}
                    >
                      {feature.description}
                    </Typography>
                  </CardContent>
                </Card>
              </motion.div>
            </Grid>
          ))}
        </Grid>

        {/* Also include the creative features grid */}
        <Box sx={{ mt: 12 }}>
          <CreativeFeatures />
        </Box>
      </Container>
    </Box>
  );
};

export default FeaturesPage;
