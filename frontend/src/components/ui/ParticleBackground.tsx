import React, { useEffect, useRef } from 'react';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  opacity: number;
  delay: number;
}

const ParticleBackground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const particlesRef = useRef<Particle[]>([]);
  const animationFrameRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Debounce resize to prevent forced reflows
    let resizeTimeout: NodeJS.Timeout;
    let rafId: number | null = null;

    const resizeCanvas = () => {
      // Cancel any pending resize
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
      }

      // Use requestAnimationFrame to batch resize operations
      rafId = requestAnimationFrame(() => {
        const width = window.innerWidth;
        const height = window.innerHeight;

        // Only resize if dimensions actually changed
        if (canvas.width !== width || canvas.height !== height) {
          canvas.width = width;
          canvas.height = height;

          // Recreate particles if canvas size changed significantly
          const newParticleCount = Math.floor((width * height) / 15000);
          if (Math.abs(newParticleCount - particlesRef.current.length) > 10) {
            particlesRef.current = Array.from({ length: newParticleCount }, () => ({
              x: Math.random() * width,
              y: Math.random() * height,
              vx: (Math.random() - 0.5) * 0.3,
              vy: (Math.random() - 0.5) * 0.3,
              size: Math.random() * 1.5 + 0.5,
              opacity: Math.random() * 0.5 + 0.2,
              delay: Math.random() * 100,
            }));
          }
        }
        rafId = null;
      });
    };

    // Initial resize
    resizeCanvas();

    // Debounced resize handler
    const handleResize = () => {
      clearTimeout(resizeTimeout);
      resizeTimeout = setTimeout(resizeCanvas, 150);
    };

    window.addEventListener('resize', handleResize, { passive: true });

    // Create particles
    const particleCount = Math.floor((window.innerWidth * window.innerHeight) / 15000);
    particlesRef.current = Array.from({ length: particleCount }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.3,
      vy: (Math.random() - 0.5) * 0.3,
      size: Math.random() * 1.5 + 0.5,
      opacity: Math.random() * 0.5 + 0.2,
      delay: Math.random() * 100,
    }));

    let frame = 0;
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particlesRef.current.forEach((particle) => {
        if (frame < particle.delay) return;

        // Update position
        particle.x += particle.vx;
        particle.y += particle.vy;

        // Wrap around edges
        if (particle.x < 0) particle.x = canvas.width;
        if (particle.x > canvas.width) particle.x = 0;
        if (particle.y < 0) particle.y = canvas.height;
        if (particle.y > canvas.height) particle.y = 0;

        // Draw particle
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0, 255, 255, ${particle.opacity})`;
        ctx.shadowBlur = 10;
        ctx.shadowColor = 'rgba(0, 255, 255, 0.8)';
        ctx.fill();
        ctx.shadowBlur = 0;

        // Draw connections to nearby particles
        particlesRef.current.forEach((other) => {
          if (other === particle) return;

          const dx = particle.x - other.x;
          const dy = particle.y - other.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 120) {
            ctx.beginPath();
            ctx.moveTo(particle.x, particle.y);
            ctx.lineTo(other.x, other.y);
            ctx.strokeStyle = `rgba(0, 255, 255, ${(1 - distance / 120) * 0.1})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        });
      });

      frame++;
      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', handleResize);
      clearTimeout(resizeTimeout);
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 1,
      }}
    />
  );
};

export default ParticleBackground;
