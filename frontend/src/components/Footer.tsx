import React, { useRef, useEffect } from 'react';
import { Box, Container, Typography, Grid, Link, IconButton, Divider } from '@mui/material';
import { motion } from 'framer-motion';
import {
  GitHub,
  Twitter,
  LinkedIn,
  Email,
  Code,
  Science,
  Security,
} from '@mui/icons-material';

const Footer: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = canvas.offsetWidth * 2;
      canvas.height = canvas.offsetHeight * 2;
      ctx.scale(2, 2);
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // AI Nodes (points on sphere)
    const numNodes = 350;
    const nodes: Array<{ x: number; y: number; z: number; connections: number[] }> = [];
    const radius = 140;

    // Generate nodes on sphere surface
    for (let i = 0; i < numNodes; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(Math.random() * 2 - 1);
      const x = radius * Math.sin(phi) * Math.cos(theta);
      const y = radius * Math.sin(phi) * Math.sin(theta);
      const z = radius * Math.cos(phi);

      nodes.push({
        x,
        y,
        z,
        connections: [],
      });
    }

    // Find connections (nodes within certain distance)
    const connectionDistance = radius * 0.25;
    nodes.forEach((node, i) => {
      nodes.forEach((other, j) => {
        if (i !== j && Math.random() > 0.7) {
          const dx = node.x - other.x;
          const dy = node.y - other.y;
          const dz = node.z - other.z;
          const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
          if (dist < connectionDistance) {
            node.connections.push(j);
          }
        }
      });
    });

    // Rotation variables
    let rotationX = 0;
    let rotationY = 0;
    const rotationSpeedX = 0.0015;
    const rotationSpeedY = 0.001;

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width / 2, canvas.height / 2);

      rotationX += rotationSpeedX;
      rotationY += rotationSpeedY;

      const centerX = canvas.width / 4;
      const centerY = canvas.height / 4;

      // Project 3D to 2D
      const projectedNodes = nodes.map((node) => {
        // Rotate around X axis
        let x = node.x;
        let y = node.y * Math.cos(rotationX) - node.z * Math.sin(rotationX);
        let z = node.y * Math.sin(rotationX) + node.z * Math.cos(rotationX);

        // Rotate around Y axis
        const tempX = x * Math.cos(rotationY) - z * Math.sin(rotationY);
        const tempZ = x * Math.sin(rotationY) + z * Math.cos(rotationY);
        x = tempX;
        z = tempZ;

        // Project to 2D
        const scale = 250 / (250 + z);
        const px = centerX + x * scale;
        const py = centerY + y * scale;

        return { px, py, z, scale };
      });

      // Draw connections
      projectedNodes.forEach((projected, i) => {
        const node = nodes[i];
        node.connections.forEach((connIndex) => {
          const connected = projectedNodes[connIndex];
          if (connected.z > -180) {
            const alpha = Math.min(projected.scale, connected.scale) * 0.2;
            ctx.strokeStyle = `hsla(200, 100%, 55%, ${alpha})`;
            ctx.lineWidth = 0.8;
            ctx.beginPath();
            ctx.moveTo(projected.px, projected.py);
            ctx.lineTo(connected.px, connected.py);
            ctx.stroke();
          }
        });
      });

      // Draw nodes
      projectedNodes.forEach((projected, i) => {
        if (projected.z > -180) {
          const size = 2.5 * projected.scale;
          const alpha = Math.min(projected.scale, 1);

          // Outer glow
          const gradient = ctx.createRadialGradient(
            projected.px,
            projected.py,
            0,
            projected.px,
            projected.py,
            size * 4
          );
          gradient.addColorStop(0, `hsla(200, 100%, 55%, ${alpha * 0.9})`);
          gradient.addColorStop(0.5, `hsla(200, 100%, 55%, ${alpha * 0.5})`);
          gradient.addColorStop(1, `hsla(200, 100%, 55%, 0)`);

          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(projected.px, projected.py, size * 4, 0, Math.PI * 2);
          ctx.fill();

          // Node center
          ctx.fillStyle = `hsla(200, 100%, 55%, ${alpha})`;
          ctx.beginPath();
          ctx.arc(projected.px, projected.py, size, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
    };
  }, []);

  const footerLinks = {
    product: [
      { label: 'Features', href: '/features' },
      { label: 'Analyze', href: '/analyze' },
      { label: 'About', href: '/about' },
    ],
    resources: [
      { label: 'Documentation', href: '#' },
      { label: 'API Reference', href: '#' },
      { label: 'GitHub', href: '#' },
    ],
    company: [
      { label: 'Privacy Policy', href: '#' },
      { label: 'Terms of Service', href: '#' },
      { label: 'Contact', href: '#' },
    ],
  };

  const socialLinks = [
    { icon: <GitHub />, href: '#', label: 'GitHub' },
    { icon: <Twitter />, href: '#', label: 'Twitter' },
    { icon: <LinkedIn />, href: '#', label: 'LinkedIn' },
    { icon: <Email />, href: '#', label: 'Email' },
  ];

  return (
    <Box
      component="footer"
      sx={{
        position: 'relative',
        mt: 'auto',
        pt: 12,
        pb: 6,
        overflow: 'hidden',
        background: 'linear-gradient(180deg, transparent 0%, hsla(0, 0%, 3%, 0.8) 30%, hsla(0, 0%, 3%, 0.95) 100%)',
        borderTop: '1px solid hsla(200, 100%, 55%, 0.2)',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '1px',
          background: 'linear-gradient(90deg, transparent, hsl(200, 100%, 55%), transparent)',
          opacity: 0.5,
        },
      }}
    >
      {/* Rotating Globe Background */}
      <Box
        sx={{
          position: 'absolute',
          bottom: '-80px',
          left: '50%',
          transform: 'translateX(-50%)',
          width: '600px',
          height: '550px',
          opacity: 0.35,
          pointerEvents: 'none',
          zIndex: 0,
          filter: 'blur(1px)',
        }}
      >
        <canvas
          ref={canvasRef}
          style={{
            width: '100%',
            height: '100%',
            display: 'block',
          }}
        />
      </Box>

      <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 10 }}>
        <Grid container spacing={6} sx={{ mb: 6 }}>
          {/* Brand Column */}
          <Grid item xs={12} md={4}>
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6 }}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
                <Box
                  sx={{
                    width: '48px',
                    height: '48px',
                    borderRadius: '12px',
                    background: 'linear-gradient(135deg, hsl(200, 100%, 55%) 0%, hsl(200, 100%, 55%) 100%)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    boxShadow: '0 0 20px hsla(200, 100%, 55%, 0.4)',
                  }}
                >
                  <Security sx={{ fontSize: 28, color: 'hsl(220, 40%, 8%)' }} />
                </Box>
                <Typography
                  variant="h5"
                  sx={{
                    fontWeight: 800,
                    fontSize: '1.5rem',
                    letterSpacing: '-0.02em',
                    background: 'linear-gradient(135deg, hsl(200, 100%, 55%) 0%, hsl(200, 100%, 55%) 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    backgroundClip: 'text',
                  }}
                >
                  DEEPFAKE DETECTION
                </Typography>
              </Box>

              <Typography
                variant="body2"
                color="text.secondary"
                sx={{
                  mb: 3,
                  lineHeight: 1.7,
                  maxWidth: '300px',
                }}
              >
                Advanced AI-powered deepfake detection with frame-by-frame analysis,
                confidence visualization, and explainable AI features.
              </Typography>

              {/* Social Links */}
              <Box sx={{ display: 'flex', gap: 1 }}>
                {socialLinks.map((social, index) => (
                  <motion.div
                    key={social.label}
                    initial={{ opacity: 0, scale: 0 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <IconButton
                      href={social.href}
                      sx={{
                        color: 'text.secondary',
                        border: '1px solid hsla(200, 100%, 55%, 0.2)',
                        borderRadius: '8px',
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          color: 'hsl(200, 100%, 55%)',
                          borderColor: 'hsl(200, 100%, 55%)',
                          boxShadow: '0 0 15px hsla(200, 100%, 55%, 0.3)',
                          background: 'hsla(200, 100%, 55%, 0.1)',
                        },
                      }}
                    >
                      {social.icon}
                    </IconButton>
                  </motion.div>
                ))}
              </Box>
            </motion.div>
          </Grid>

          {/* Links Columns */}
          <Grid item xs={12} md={8}>
            <Grid container spacing={4}>
              {Object.entries(footerLinks).map(([category, links], categoryIndex) => (
                <Grid item xs={6} sm={4} key={category}>
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5, delay: categoryIndex * 0.1 }}
                  >
                    <Typography
                      variant="h6"
                      sx={{
                        mb: 2,
                        fontWeight: 700,
                        fontSize: '0.9rem',
                        letterSpacing: '0.1em',
                        textTransform: 'uppercase',
                        color: 'hsl(200, 100%, 55%)',
                      }}
                    >
                      {category}
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                      {links.map((link, linkIndex) => (
                        <motion.div
                          key={link.label}
                          initial={{ opacity: 0, x: -10 }}
                          whileInView={{ opacity: 1, x: 0 }}
                          viewport={{ once: true }}
                          transition={{ duration: 0.4, delay: categoryIndex * 0.1 + linkIndex * 0.05 }}
                        >
                          <Link
                            href={link.href}
                            color="text.secondary"
                            sx={{
                              textDecoration: 'none',
                              fontSize: '0.9rem',
                              display: 'flex',
                              alignItems: 'center',
                              gap: 1,
                              transition: 'all 0.3s ease',
                              position: 'relative',
                              '&::before': {
                                content: '""',
                                position: 'absolute',
                                left: 0,
                                width: '0',
                                height: '1px',
                                background: 'hsl(200, 100%, 55%)',
                                transition: 'width 0.3s ease',
                              },
                              '&:hover': {
                                color: 'hsl(200, 100%, 55%)',
                                paddingLeft: '8px',
                                '&::before': {
                                  width: '4px',
                                },
                              },
                            }}
                          >
                            {link.label}
                          </Link>
                        </motion.div>
                      ))}
                    </Box>
                  </motion.div>
                </Grid>
              ))}
            </Grid>
          </Grid>
        </Grid>

        {/* Bottom Section */}
        <Divider sx={{ borderColor: 'hsla(200, 100%, 55%, 0.2)', mb: 4 }} />

        <Box
          sx={{
            display: 'flex',
            flexDirection: { xs: 'column', md: 'row' },
            justifyContent: 'space-between',
            alignItems: 'center',
            gap: 3,
          }}
        >
          <Typography
            variant="body2"
            color="text.secondary"
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
            }}
          >
            <Code sx={{ fontSize: 16 }} />
            Built with advanced neural networks
          </Typography>

          <Typography
            variant="body2"
            color="text.secondary"
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 1,
            }}
          >
            <Science sx={{ fontSize: 16 }} />
            Powered by AI Technology
          </Typography>

          <Typography variant="body2" color="text.secondary">
            © {new Date().getFullYear()} Deepfake Detection System. All rights reserved.
          </Typography>
        </Box>
      </Container>
    </Box>
  );
};

export default Footer;
