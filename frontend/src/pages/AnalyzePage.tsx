import React from 'react';
import { Box, Container } from '@mui/material';
import { motion } from 'framer-motion';
import EnhancedVideoAnalysis from '../components/EnhancedVideoAnalysis';

const AnalyzePage: React.FC = () => {
  return (
    <Box
      sx={{
        minHeight: 'calc(100vh - 64px)',
        py: { xs: 4, md: 8 },
        position: 'relative',
        overflow: 'visible',
      }}
    >
      {/* Animated Background Elements */}
      <Box
        sx={{
          position: 'absolute',
          top: '10%',
          right: '5%',
          width: '400px',
          height: '400px',
          borderRadius: '50%',
          background: 'radial-gradient(circle, hsla(200, 100%, 55%, 0.1) 0%, transparent 70%)',
          filter: 'blur(80px)',
          zIndex: 0,
          animation: 'pulse 6s ease-in-out infinite',
        }}
      />
      <Box
        sx={{
          position: 'absolute',
          bottom: '20%',
          left: '10%',
          width: '300px',
          height: '300px',
          borderRadius: '50%',
          background: 'radial-gradient(circle, hsla(300, 100%, 60%, 0.08) 0%, transparent 60%)',
          filter: 'blur(60px)',
          zIndex: 0,
          animation: 'pulse 8s ease-in-out infinite',
          animationDelay: '1s',
        }}
      />

      <Container maxWidth="xl" sx={{ position: 'relative', zIndex: 10 }}>
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <EnhancedVideoAnalysis />
        </motion.div>
      </Container>
    </Box>
  );
};

export default AnalyzePage;
