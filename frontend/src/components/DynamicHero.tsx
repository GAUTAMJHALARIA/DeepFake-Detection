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
      }}
    >
      {/* Animated Background Orbs */}
      <Box
        sx={{
          position: 'absolute',
          width: '600px',
          height: '600px',
          borderRadius: '50%',
          background: 'radial-gradient(circle, hsl(200, 100%, 55%, 0.15) 0%, transparent 70%)',
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
          background: 'radial-gradient(circle, hsl(300, 100%, 60%, 0.12) 0%, transparent 70%)',
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
                    mb: 2,
                    color: 'hsl(200, 100%, 55%)',
                    fontSize: '0.9rem',
                    letterSpacing: '0.3em',
                    fontWeight: 600,
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
                    fontSize: { xs: '3rem', md: '5rem', lg: '6.5rem' },
                    fontWeight: 900,
                    lineHeight: 1.15,
                    mb: 3,
                    letterSpacing: '-0.04em',
                    position: 'relative',
                    textTransform: 'uppercase',
                    overflow: 'visible',
                    wordBreak: 'normal',
                    whiteSpace: 'normal',
                    display: 'block',
                    width: '100%',
                  }}
                >
                  <Box
                    component="span"
                    sx={{
                      display: 'block',
                      background: 'linear-gradient(135deg, hsl(200, 100%, 55%) 0%, hsl(200, 100%, 55%) 30%, hsl(300, 100%, 60%) 60%, hsl(200, 100%, 55%) 100%)',
                      backgroundSize: '200% auto',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      backgroundClip: 'text',
                      animation: 'gradientText 4s ease infinite',
                      mb: 1,
                    }}
                  >
                    DETECT
                  </Box>
                  <Box
                    component="span"
                    sx={{
                      display: 'block',
                      background: 'linear-gradient(135deg, hsl(200, 100%, 55%) 0%, hsl(200, 100%, 55%) 30%, hsl(300, 100%, 60%) 60%, hsl(200, 100%, 55%) 100%)',
                      backgroundSize: '200% auto',
                      WebkitBackgroundClip: 'text',
                      WebkitTextFillColor: 'transparent',
                      backgroundClip: 'text',
                      animation: 'gradientText 4s ease infinite',
                      position: 'relative',
                      width: '100%',
                      overflow: 'visible',
                      '&::after': {
                        content: '""',
                        position: 'absolute',
                        left: 0,
                        bottom: '0.1em',
                        width: '100%',
                        height: '0.15em',
                        background: 'linear-gradient(90deg, hsl(200, 100%, 55%), hsl(300, 100%, 60%))',
                        opacity: 0.3,
                        filter: 'blur(8px)',
                      },
                    }}
                  >
                    DEEPFAKES
                  </Box>
                </Typography>
              </motion.div>

              <motion.div variants={itemVariants}>
                <Typography
                  variant="h5"
                  sx={{
                    color: 'text.secondary',
                    mb: 4,
                    lineHeight: 1.6,
                    fontSize: { xs: '1.1rem', md: '1.4rem' },
                    maxWidth: '600px',
                    fontWeight: 300,
                    letterSpacing: '0.02em',
                  }}
                >
                  Uncover synthetic media with{' '}
                  <Box
                    component="span"
                    sx={{
                      color: 'hsl(200, 100%, 55%)',
                      fontWeight: 600,
                      textShadow: '0 0 20px hsla(200, 100%, 55%, 0.5)',
                    }}
                  >
                    AI-powered precision
                  </Box>
                  . Frame-by-frame analysis, explainable insights, and real-time confidence scoring.
                </Typography>
              </motion.div>

              <motion.div variants={itemVariants}>
                <Box sx={{ display: 'flex', gap: 3, flexWrap: 'wrap', mb: 4 }}>
                  <motion.div
                    whileHover={{ scale: 1.05, y: -2 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Button
                      variant="contained"
                      size="large"
                      onClick={onGetStarted}
                      startIcon={<PlayArrow />}
                      sx={{
                        px: 5,
                        py: 1.8,
                        fontSize: '1.1rem',
                        fontWeight: 700,
                        textTransform: 'uppercase',
                        letterSpacing: '0.15em',
                        background: 'linear-gradient(135deg, hsl(200, 100%, 55%) 0%, hsl(200, 100%, 55%) 100%)',
                        color: 'hsl(220, 40%, 8%)',
                        boxShadow: '0 0 40px hsla(200, 100%, 55%, 0.5), 0 10px 30px rgba(0,0,0,0.3)',
                        borderRadius: '12px',
                        border: '2px solid transparent',
                        position: 'relative',
                        overflow: 'hidden',
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: '-100%',
                          width: '100%',
                          height: '100%',
                          background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)',
                          transition: 'left 0.5s',
                        },
                        '&:hover::before': {
                          left: '100%',
                        },
                        '&:hover': {
                          background: 'linear-gradient(135deg, hsl(180, 100%, 55%) 0%, hsl(200, 100%, 60%) 100%)',
                          boxShadow: '0 0 60px hsla(200, 100%, 55%, 0.7), 0 15px 40px rgba(0,0,0,0.4)',
                          transform: 'translateY(-2px)',
                        },
                      }}
                    >
                      Start Analysis
                    </Button>
                  </motion.div>

                  <motion.div
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Button
                      variant="outlined"
                      size="large"
                      sx={{
                        px: 4,
                        py: 1.8,
                        fontSize: '1.1rem',
                        fontWeight: 600,
                        textTransform: 'uppercase',
                        letterSpacing: '0.15em',
                        borderWidth: '2px',
                        borderColor: 'hsl(300, 100%, 60%)',
                        color: 'hsl(300, 100%, 60%)',
                        borderRadius: '12px',
                        position: 'relative',
                        overflow: 'hidden',
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          width: '0%',
                          height: '100%',
                          background: 'hsla(300, 100%, 60%, 0.1)',
                          transition: 'width 0.3s',
                        },
                        '&:hover::before': {
                          width: '100%',
                        },
                        '&:hover': {
                          borderColor: 'hsl(300, 100%, 65%)',
                          color: 'hsl(300, 100%, 65%)',
                          boxShadow: '0 0 30px hsla(300, 100%, 60%, 0.4)',
                        },
                      }}
                    >
                      Explore Features
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
                  color: 'hsl(200, 100%, 55%)',
                  fontSize: '2rem',
                  filter: 'drop-shadow(0 0 10px hsl(200, 100%, 55%))',
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
