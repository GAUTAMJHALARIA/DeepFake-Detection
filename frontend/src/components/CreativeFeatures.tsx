import React, { useState } from 'react';
import { Box, Typography, Container, Card, CardContent } from '@mui/material';
import {
  CloudUpload,
  VideoFile,
  Timeline,
  BarChart,
  Psychology,
  Speed,
} from '@mui/icons-material';
import { motion } from 'framer-motion';

const features = [
  {
    icon: <CloudUpload sx={{ fontSize: 50 }} />,
    title: 'Drag & Drop',
    description: 'Upload videos instantly with intuitive drag-and-drop interface',
    color: 'hsl(200, 100%, 55%)',
    position: { gridColumn: 'span 2', gridRow: 'span 2' },
  },
  {
    icon: <Timeline sx={{ fontSize: 40 }} />,
    title: 'Frame Analysis',
    description: 'Deep frame-by-frame inspection',
    color: 'hsl(200, 100%, 55%)',
    position: { gridColumn: 'span 1', gridRow: 'span 1' },
  },
  {
    icon: <BarChart sx={{ fontSize: 40 }} />,
    title: 'Heat Maps',
    description: 'Interactive confidence visualization',
    color: 'hsl(300, 100%, 60%)',
    position: { gridColumn: 'span 1', gridRow: 'span 1' },
  },
  {
    icon: <Psychology sx={{ fontSize: 40 }} />,
    title: 'Grad-CAM',
    description: 'Explainable AI insights',
    color: 'hsl(200, 100%, 55%)',
    position: { gridColumn: 'span 1', gridRow: 'span 1' },
  },
  {
    icon: <Speed sx={{ fontSize: 40 }} />,
    title: 'Real-Time',
    description: 'Instant processing results',
    color: 'hsl(200, 100%, 55%)',
    position: { gridColumn: 'span 1', gridRow: 'span 1' },
  },
  {
    icon: <VideoFile sx={{ fontSize: 40 }} />,
    title: 'Multi-Format',
    description: 'Supports all major video formats',
    color: 'hsl(300, 100%, 60%)',
    position: { gridColumn: 'span 2', gridRow: 'span 1' },
  },
];

const CreativeFeatures: React.FC = () => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  return (
    <Box
      sx={{
        py: { xs: 10, md: 15 },
        position: 'relative',
        overflow: 'hidden',
      }}
      id="features"
    >
      <Container maxWidth="xl">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
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
              POWERFUL CAPABILITIES
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
              Everything You Need
            </Typography>
          </Box>
        </motion.div>

        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', md: 'repeat(4, 1fr)' },
            gridAutoRows: 'minmax(200px, auto)',
            gap: 3,
          }}
        >
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, scale: 0.8 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              whileHover={{ scale: 1.05, zIndex: 10 }}
              onHoverStart={() => setHoveredIndex(index)}
              onHoverEnd={() => setHoveredIndex(null)}
              style={{
                ...feature.position,
                gridColumn: feature.position.gridColumn,
                gridRow: feature.position.gridRow,
              }}
            >
              <Card
                sx={{
                  height: '100%',
                  background: hoveredIndex === index
                    ? `linear-gradient(135deg, ${feature.color}15 0%, ${feature.color}08 100%)`
                    : 'linear-gradient(135deg, hsla(220, 35%, 12%, 0.9) 0%, hsla(220, 30%, 15%, 0.8) 100%)',
                  backdropFilter: 'blur(20px)',
                  border: `2px solid ${hoveredIndex === index ? feature.color : 'hsla(200, 100%, 55%, 0.2)'}`,
                  borderRadius: '20px',
                  position: 'relative',
                  overflow: 'hidden',
                  transition: 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)',
                  cursor: 'pointer',
                  '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: '-50%',
                    left: '-50%',
                    width: '200%',
                    height: '200%',
                    background: `radial-gradient(circle, ${feature.color}20 0%, transparent 70%)`,
                    opacity: hoveredIndex === index ? 1 : 0,
                    transition: 'opacity 0.4s',
                  },
                  '&::after': {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    border: `2px solid ${feature.color}`,
                    borderRadius: '20px',
                    opacity: 0,
                    transition: 'opacity 0.4s',
                    boxShadow: hoveredIndex === index ? `0 0 40px ${feature.color}88` : 'none',
                  },
                  '&:hover::after': {
                    opacity: hoveredIndex === index ? 0.6 : 0,
                  },
                }}
              >
                <CardContent
                  sx={{
                    p: { xs: 3, md: 4 },
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center',
                    position: 'relative',
                    zIndex: 1,
                  }}
                >
                  <motion.div
                    animate={{
                      scale: hoveredIndex === index ? [1, 1.2, 1] : 1,
                      rotate: hoveredIndex === index ? [0, 5, -5, 0] : 0,
                    }}
                    transition={{ duration: 0.5 }}
                  >
                    <Box
                      sx={{
                        color: feature.color,
                        mb: 2,
                        filter: hoveredIndex === index
                          ? `drop-shadow(0 0 20px ${feature.color})`
                          : `drop-shadow(0 0 10px ${feature.color}88)`,
                        transition: 'all 0.3s',
                      }}
                    >
                      {feature.icon}
                    </Box>
                  </motion.div>

                  <Typography
                    variant="h5"
                    sx={{
                      mb: 2,
                      fontWeight: 700,
                      fontSize: { xs: '1.3rem', md: '1.6rem' },
                      color: hoveredIndex === index ? feature.color : 'text.primary',
                      transition: 'color 0.3s',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                    }}
                  >
                    {feature.title}
                  </Typography>

                  <Typography
                    variant="body2"
                    sx={{
                      color: 'text.secondary',
                      lineHeight: 1.7,
                      fontSize: { xs: '0.9rem', md: '1rem' },
                      opacity: hoveredIndex === index ? 1 : 0.8,
                      transition: 'opacity 0.3s',
                    }}
                  >
                    {feature.description}
                  </Typography>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </Box>
      </Container>
    </Box>
  );
};

export default CreativeFeatures;
