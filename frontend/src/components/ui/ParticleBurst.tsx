import React, { useEffect, useState } from 'react';
import { Box } from '@mui/material';
import { motion } from 'framer-motion';

interface ParticleBurstProps {
  trigger: boolean;
  onComplete?: () => void;
}

const ParticleBurst: React.FC<ParticleBurstProps> = ({ trigger, onComplete }) => {
  const [particles, setParticles] = useState<Array<{ id: number; x: number; y: number; angle: number; color: string }>>([]);

  useEffect(() => {
    if (trigger) {
      // Generate particles in a burst pattern
      const newParticles = [];
      const numParticles = 50;
      const centerX = 50; // Percentage
      const centerY = 50; // Percentage

      for (let i = 0; i < numParticles; i++) {
        const angle = (i / numParticles) * Math.PI * 2;
        const speed = 0.5 + Math.random() * 0.5;
        const distance = 30 + Math.random() * 20;

        const colors = [
          'hsl(200, 100%, 55%)',
          'hsl(200, 100%, 55%)',
          'hsl(300, 100%, 60%)',
          'hsl(180, 100%, 60%)',
        ];

        newParticles.push({
          id: i,
          x: centerX,
          y: centerY,
          angle: angle,
          color: colors[Math.floor(Math.random() * colors.length)],
        });
      }

      setParticles(newParticles);

      // Trigger completion callback after animation
      if (onComplete) {
        setTimeout(() => {
          onComplete();
        }, 2000);
      }
    }
  }, [trigger, onComplete]);

  if (!trigger || particles.length === 0) return null;

  return (
    <Box
      sx={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 9999,
      }}
    >
      {particles.map((particle) => (
        <motion.div
          key={particle.id}
          initial={{
            x: `${particle.x}%`,
            y: `${particle.y}%`,
            opacity: 1,
            scale: 1,
          }}
          animate={{
            x: `${particle.x + Math.cos(particle.angle) * 30}%`,
            y: `${particle.y + Math.sin(particle.angle) * 30}%`,
            opacity: 0,
            scale: 0,
          }}
          transition={{
            duration: 2,
            ease: 'easeOut',
          }}
          style={{
            position: 'absolute',
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: particle.color,
            boxShadow: `0 0 20px ${particle.color}, 0 0 40px ${particle.color}`,
          }}
        />
      ))}
    </Box>
  );
};

export default ParticleBurst;
