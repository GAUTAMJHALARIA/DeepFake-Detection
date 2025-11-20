import React, { useState, useEffect } from 'react';
import { Box, Typography, Button, Container } from '@mui/material';
import { PlayArrow, KeyboardArrowDown } from '@mui/icons-material';
import { motion, useAnimation, useInView } from 'framer-motion';
import { SplineSceneBasic } from './ui/demo';

interface DynamicHeroProps {
  onGetStarted: () => void;
}

const DynamicHero: React.FC<DynamicHeroProps> = ({ onGetStarted }) => {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const controls = useAnimation();
  const ref = React.useRef(null);
  const inView = useInView(ref);

  // Generate random dots positions once
  const [dots] = useState(() =>
    Array.from({ length: 20 }, () => ({
      left: Math.random() * 100,
      top: Math.random() * 100,
      delay: Math.random() * 2,
    }))
  );

  useEffect(() => {
    if (inView) {
      controls.start('visible');
    }
  }, [controls, inView]);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({
        x: (e.clientX / window.innerWidth - 0.5) * 20,
        y: (e.clientY / window.innerHeight - 0.5) * 20,
      });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.3,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 50, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        type: 'spring' as const,
        stiffness: 100,
        damping: 12,
      },
    },
  };

  return (
    <Box
      ref={ref}
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
        overflow: 'visible',
        pt: { xs: 12, md: 8 },
        pb: { xs: 8, md: 12 },
        background: '#0A0F17',
        // Subtle grid pattern overlay
        backgroundImage: `
          linear-gradient(rgba(0, 255, 255, 0.03) 1px, transparent 1px),
          linear-gradient(90deg, rgba(0, 255, 255, 0.03) 1px, transparent 1px)
        `,
        backgroundSize: '50px 50px',
      }}
    >
      {/* Glowing teal/cyan dots scattered across background */}
      {dots.map((dot, i) => (
        <Box
          key={i}
          sx={{
            position: 'absolute',
            width: '4px',
            height: '4px',
            borderRadius: '50%',
            background: '#00FFFF',
            left: `${dot.left}%`,
            top: `${dot.top}%`,
            boxShadow: '0 0 10px #00FFFF, 0 0 20px #00CED1',
            opacity: 0.6,
            animation: 'pulse 3s ease-in-out infinite',
            animationDelay: `${dot.delay}s`,
            zIndex: 0,
          }}
        />
      ))}

      {/* Animated Background Orbs */}
      <Box
        sx={{
          position: 'absolute',
          width: '600px',
          height: '600px',
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(0, 255, 255, 0.1) 0%, transparent 70%)',
          left: `calc(20% + ${mousePosition.x * 2}px)`,
          top: `calc(20% + ${mousePosition.y * 2}px)`,
          filter: 'blur(60px)',
          animation: 'pulse 4s ease-in-out infinite',
          zIndex: 0,
        }}
      />
      <Box
        sx={{
          position: 'absolute',
          width: '500px',
          height: '500px',
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(0, 206, 209, 0.08) 0%, transparent 70%)',
          right: `calc(15% - ${mousePosition.x * 1.5}px)`,
          bottom: `calc(15% - ${mousePosition.y * 1.5}px)`,
          filter: 'blur(50px)',
          animation: 'pulse 5s ease-in-out infinite',
          animationDelay: '1s',
          zIndex: 0,
        }}
      />

      <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 10 }}>
        <motion.div
          variants={containerVariants}
          initial="hidden"
          animate={controls}
          style={{ width: '100%' }}
        >
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', lg: '1fr 1fr' },
              gap: { xs: 4, lg: 6 },
              alignItems: 'center',
              minHeight: { lg: '600px' },
              overflow: 'visible',
              width: '100%',
            }}
          >
            {/* Left: Text Content */}
            <Box
              sx={{
                overflow: 'visible',
                position: 'relative',
                zIndex: 10,
                width: '100%',
              }}
            >
              <motion.div variants={itemVariants}>
                <Typography
                  variant="overline"
                  sx={{
                    display: 'block',
                    mb: 2.5,
                    color: '#66A3FF',
                    fontSize: { xs: '0.75rem', md: '0.875rem' },
                    letterSpacing: '0.2em',
                    fontWeight: 500,
                    textTransform: 'uppercase',
                  }}
                >
                  NEXT-GEN DETECTION
                </Typography>
              </motion.div>

              <motion.div variants={itemVariants}>
                <Typography
                  variant="h1"
                  component="div"
                  sx={{
                    fontSize: { xs: '2.5rem', md: '4.5rem', lg: '5.5rem' },
                    fontWeight: 800,
                    lineHeight: 1.1,
                    mb: 3.5,
                    letterSpacing: '-0.02em',
                    position: 'relative',
                    textTransform: 'uppercase',
                    overflow: 'visible',
                    wordBreak: 'normal',
                    whiteSpace: 'normal',
                    display: 'block',
                    width: '100%',
                    color: '#FFFFFF',
                  }}
                >
                  <Box
                    component="span"
                    sx={{
                      display: 'block',
                      color: '#FFFFFF',
                      mb: 0.5,
                    }}
                  >
                    DETECT
                  </Box>
                  <Box
                    component="span"
                    sx={{
                      display: 'block',
                      color: '#FFFFFF',
                      position: 'relative',
                      width: '100%',
                      overflow: 'visible',
                    }}
                  >
                    DEEPFAKES
                  </Box>
                </Typography>
              </motion.div>

              <motion.div variants={itemVariants}>
                <Typography
                  variant="body1"
                  sx={{
                    color: '#B0B0B0',
                    mb: 4.5,
                    lineHeight: 1.7,
                    fontSize: { xs: '1rem', md: '1.125rem' },
                    maxWidth: '580px',
                    fontWeight: 400,
                    letterSpacing: '0.01em',
                  }}
                >
                  Uncover synthetic media with{' '}
                  <Box
                    component="span"
                    sx={{
                      color: '#4285F4',
                      fontWeight: 500,
                    }}
                  >
                    AI-powered precision
                  </Box>
                  . Frame-by-frame analysis, explainable insights, and real-time confidence scoring.
                </Typography>
              </motion.div>

              <motion.div variants={itemVariants}>
                <Box
                  sx={{
                    display: 'flex',
                    gap: 2.5,
                    flexWrap: { xs: 'wrap', sm: 'nowrap' },
                    mb: 4,
                    alignItems: 'center',
                    flexDirection: { xs: 'column', sm: 'row' },
                  }}
                >
                  <motion.div
                    whileHover={{ scale: 1.03, y: -2 }}
                    whileTap={{ scale: 0.97 }}
                  >
                    <Button
                      variant="contained"
                      size="large"
                      onClick={onGetStarted}
                      startIcon={<PlayArrow sx={{ color: '#FFFFFF', fontSize: '1.25rem' }} />}
                      sx={{
                        px: { xs: 4, md: 5 },
                        py: { xs: 1.5, md: 1.75 },
                        fontSize: { xs: '0.875rem', md: '0.9375rem' },
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        letterSpacing: '0.1em',
                        background: 'linear-gradient(90deg, #007BFF 0%, #4285F4 100%)',
                        color: '#FFFFFF',
                        boxShadow: '0 0 40px rgba(66, 133, 244, 0.5), 0 10px 30px rgba(0,0,0,0.3)',
                        borderRadius: '8px',
                        border: 'none',
                        position: 'relative',
                        overflow: 'hidden',
                        minWidth: { xs: '200px', md: '220px' },
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: '-100%',
                          width: '100%',
                          height: '100%',
                          background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent)',
                          transition: 'left 0.5s',
                        },
                        '&:hover::before': {
                          left: '100%',
                        },
                        '&:hover': {
                          background: 'linear-gradient(90deg, #0056B3 0%, #3399FF 100%)',
                          boxShadow: '0 0 60px rgba(66, 133, 244, 0.7), 0 15px 40px rgba(0,0,0,0.4)',
                          transform: 'translateY(-2px)',
                        },
                      }}
                    >
                      START ANALYSIS
                    </Button>
                  </motion.div>

                  <motion.div
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                  >
                    <Button
                      variant="outlined"
                      size="large"
                      sx={{
                        px: { xs: 4, md: 4.5 },
                        py: { xs: 1.5, md: 1.75 },
                        fontSize: { xs: '0.875rem', md: '0.9375rem' },
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        letterSpacing: '0.1em',
                        borderWidth: '2px',
                        borderColor: '#FFFFFF',
                        color: '#FFFFFF',
                        background: 'transparent',
                        borderRadius: '8px',
                        position: 'relative',
                        overflow: 'hidden',
                        minWidth: { xs: '200px', md: '220px' },
                        '&:hover': {
                          borderColor: '#FFFFFF',
                          color: '#FFFFFF',
                          background: 'rgba(255, 255, 255, 0.08)',
                          boxShadow: '0 0 30px rgba(255, 255, 255, 0.15)',
                        },
                      }}
                    >
                      EXPLORE FEATURES
                    </Button>
                  </motion.div>
                </Box>
              </motion.div>

            </Box>

            {/* Right: Spline Robot Scene - Seamlessly Integrated with Background */}
            <Box
              sx={{
                display: { xs: 'none', lg: 'flex' },
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative',
                height: { lg: '600px' },
                overflow: 'visible',
                width: '100%',
                minWidth: '100%',
              }}
            >
              <motion.div
                initial={{ scale: 0.8, opacity: 0, x: 50 }}
                animate={{ scale: 1, opacity: 1, x: 0 }}
                transition={{
                  type: 'spring',
                  stiffness: 100,
                  damping: 15,
                  delay: 0.5,
                }}
                style={{
                  width: '100%',
                  height: '100%',
                  position: 'relative',
                }}
              >
                {/* Subtle Radial Glow Behind Robot - Very Subtle */}
                <Box
                  sx={{
                    position: 'absolute',
                    top: '50%',
                    left: '60%',
                    transform: 'translate(-50%, -50%)',
                    width: '700px',
                    height: '700px',
                    borderRadius: '50%',
                    background: 'radial-gradient(circle, hsla(200, 100%, 55%, 0.08) 0%, transparent 60%)',
                    filter: 'blur(80px)',
                    zIndex: 1,
                    animation: 'pulse 5s ease-in-out infinite',
                  }}
                />

                {/* Additional Ambient Glow */}
                <Box
                  sx={{
                    position: 'absolute',
                    top: '40%',
                    left: '60%',
                    transform: 'translate(-50%, -50%)',
                    width: '500px',
                    height: '500px',
                    borderRadius: '50%',
                    background: 'radial-gradient(circle, hsla(200, 100%, 55%, 0.06) 0%, transparent 50%)',
                    filter: 'blur(50px)',
                    zIndex: 1,
                    animation: 'pulse 6s ease-in-out infinite',
                    animationDelay: '1s',
                  }}
                />

                {/* Robot Scene - Directly on Background with Expanded Interaction Area */}
                <Box
                  sx={{
                    position: 'absolute',
                    top: '50%',
                    left: '60%',
                    transform: 'translate(-50%, -50%)',
                    width: '800px',
                    height: '800px',
                    zIndex: 2,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    pointerEvents: 'auto',
                    cursor: 'pointer',
                    '& canvas': {
                      borderRadius: '0 !important',
                      mixBlendMode: 'normal',
                      width: '100% !important',
                      height: '100% !important',
                    },
                    '& > div': {
                      width: '100% !important',
                      height: '100% !important',
                      background: 'transparent !important',
                      position: 'relative',
                    },
                  }}
                >
                  <SplineSceneBasic />
                </Box>
              </motion.div>
            </Box>
          </Box>

          {/* Scroll Indicator */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.5 }}
            style={{
              position: 'absolute',
              bottom: '40px',
              left: '50%',
              transform: 'translateX(-50%)',
            }}
          >
            <motion.div
              animate={{ y: [0, 10, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <KeyboardArrowDown
                sx={{
                  color: '#4285F4',
                  fontSize: '2rem',
                  filter: 'drop-shadow(0 0 10px rgba(66, 133, 244, 0.5))',
                  cursor: 'pointer',
                }}
                onClick={() => {
                  document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' });
                }}
              />
            </motion.div>
          </motion.div>
        </motion.div>
      </Container>
    </Box>
  );
};

export default DynamicHero;
