import React from 'react';
import { Box, Container, Typography, Grid, Card, CardContent, Divider } from '@mui/material';
import { motion } from 'framer-motion';
import {
  Security,
  Speed,
  Psychology,
  Analytics,
} from '@mui/icons-material';

const stats = [
  { value: '99.8%', label: 'Accuracy Rate', icon: <Analytics /> },
  { value: '0.2s', label: 'Per Frame', icon: <Speed /> },
  { value: '100+', label: 'Formats Supported', icon: <Security /> },
  { value: 'AI', label: 'Grad-CAM++', icon: <Psychology /> },
];

const AboutPage: React.FC = () => {
  return (
    <Box sx={{ minHeight: 'calc(100vh - 64px)', py: { xs: 6, md: 10 } }}>
      <Container maxWidth="lg">
        {/* Header */}
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
              ABOUT THE SYSTEM
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
              Deepfake Detection
              <br />
              Technology
            </Typography>
          </Box>
        </motion.div>

        {/* Description */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <Grid container spacing={6} sx={{ mb: 8 }}>
            <Grid item xs={12} md={6}>
              <Typography
                variant="h5"
                sx={{
                  mb: 3,
                  fontWeight: 600,
                  color: 'hsl(200, 100%, 55%)',
                }}
              >
                Advanced AI Technology
              </Typography>
              <Typography
                variant="body1"
                color="text.secondary"
                sx={{
                  lineHeight: 1.8,
                  fontSize: '1.1rem',
                  mb: 3,
                }}
              >
                Our deepfake detection system leverages state-of-the-art machine learning models
                to identify synthetic media with unprecedented accuracy. Built on advanced neural
                networks trained on diverse datasets, the system analyzes videos frame-by-frame
                to detect subtle artifacts and inconsistencies.
              </Typography>
              <Typography
                variant="body1"
                color="text.secondary"
                sx={{
                  lineHeight: 1.8,
                  fontSize: '1.1rem',
                }}
              >
                The platform combines deep learning with explainable AI techniques, providing
                not just detection results but also visual explanations through Grad-CAM++
                heatmaps showing exactly which regions influenced the model's decision.
              </Typography>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography
                variant="h5"
                sx={{
                  mb: 3,
                  fontWeight: 600,
                  color: 'hsl(200, 100%, 55%)',
                }}
              >
                Key Capabilities
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {[
                  'Real-time video analysis with frame-by-frame processing',
                  'Multi-format support (MP4, AVI, MOV, MKV, WebM)',
                  'URL-based analysis from major platforms',
                  'Interactive confidence heat maps',
                  'Explainable AI with Grad-CAM++ visualizations',
                  'Comprehensive statistical analysis',
                ].map((item, index) => (
                  <Box
                    key={index}
                    sx={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 2,
                      p: 2,
                      borderRadius: '12px',
                      background: 'hsla(220, 35%, 12%, 0.5)',
                      border: '1px solid hsla(200, 100%, 55%, 0.1)',
                    }}
                  >
                    <Box
                      sx={{
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        background: 'hsl(200, 100%, 55%)',
                        boxShadow: '0 0 10px hsl(200, 100%, 55%)',
                      }}
                    />
                    <Typography variant="body1" sx={{ flex: 1 }}>
                      {item}
                    </Typography>
                  </Box>
                ))}
              </Box>
            </Grid>
          </Grid>
        </motion.div>

        <Divider sx={{ my: 8, borderColor: 'hsla(200, 100%, 55%, 0.2)' }} />

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <Grid container spacing={4}>
            {stats.map((stat, index) => (
              <Grid item xs={6} md={3} key={index}>
                <Card
                  sx={{
                    textAlign: 'center',
                    background: 'linear-gradient(135deg, hsla(220, 35%, 12%, 0.9) 0%, hsla(220, 30%, 15%, 0.8) 100%)',
                    backdropFilter: 'blur(20px)',
                    border: '1px solid hsla(200, 100%, 55%, 0.2)',
                    borderRadius: '20px',
                    p: 4,
                    transition: 'all 0.3s ease',
                    '&:hover': {
                      borderColor: 'hsl(200, 100%, 55%)',
                      boxShadow: '0 0 30px hsla(200, 100%, 55%, 0.3)',
                      transform: 'translateY(-5px)',
                    },
                  }}
                >
                  <CardContent>
                    <Box
                      sx={{
                        color: 'hsl(200, 100%, 55%)',
                        mb: 2,
                        display: 'flex',
                        justifyContent: 'center',
                        filter: 'drop-shadow(0 0 10px hsl(200, 100%, 55%))',
                      }}
                    >
                      {stat.icon}
                    </Box>
                    <Typography
                      variant="h3"
                      sx={{
                        mb: 1,
                        background: 'linear-gradient(135deg, hsl(200, 100%, 55%) 0%, hsl(200, 100%, 55%) 100%)',
                        WebkitBackgroundClip: 'text',
                        WebkitTextFillColor: 'transparent',
                        backgroundClip: 'text',
                        fontWeight: 800,
                      }}
                    >
                      {stat.value}
                    </Typography>
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{
                        fontSize: '0.9rem',
                        letterSpacing: '0.05em',
                        textTransform: 'uppercase',
                      }}
                    >
                      {stat.label}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </motion.div>
      </Container>
    </Box>
  );
};

export default AboutPage;
