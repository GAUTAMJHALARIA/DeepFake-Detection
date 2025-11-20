'use client'

import React, { Suspense, lazy, useState, useEffect, useRef } from 'react'
import { Box, Typography } from '@mui/material'
const Spline = lazy(() => import('@splinetool/react-spline'))

interface SplineSceneProps {
  scene: string
  className?: string
}

export function SplineScene({ scene, className }: SplineSceneProps) {
  const [shouldLoad, setShouldLoad] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Use Intersection Observer to defer loading until component is visible
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setShouldLoad(true);
            observer.disconnect();
          }
        });
      },
      {
        rootMargin: '50px', // Start loading 50px before it's visible
        threshold: 0.01,
      }
    );

    observer.observe(container);

    return () => {
      observer.disconnect();
    };
  }, []);

  return (
    <Box
      ref={containerRef}
      sx={{
        width: '100%',
        height: '100%',
        minWidth: '100%',
        minHeight: '100%',
        position: 'relative',
        pointerEvents: 'auto',
      }}
    >
      {shouldLoad ? (
        <Suspense
          fallback={
            <Box
              sx={{
                width: '100%',
                height: '100%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <span className="loader"></span>
            </Box>
          }
        >
          <SplineWrapper scene={scene} />
        </Suspense>
      ) : (
        <Box
          sx={{
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <span className="loader"></span>
        </Box>
      )}
    </Box>
  );
}

// Separate component to handle WebGL errors
function SplineWrapper({ scene }: { scene: string }) {
  const [hasError, setHasError] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // WebGL errors during initialization are expected and handled gracefully
    // The Spline library handles these internally, so we just catch any unhandled errors
    const handleError = (event: ErrorEvent) => {
      if (event.message && event.message.includes('WebGL')) {
        event.preventDefault();
        setHasError(true);
      }
    };

    window.addEventListener('error', handleError);
    return () => {
      window.removeEventListener('error', handleError);
    };
  }, []);

  if (hasError) {
    return (
      <Box
        sx={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          color: 'text.secondary',
        }}
      >
        <Typography variant="body2">3D scene unavailable</Typography>
      </Box>
    );
  }

  return (
    <Box
      ref={containerRef}
      sx={{
        width: '100%',
        height: '100%',
        minWidth: '100%',
        minHeight: '100%',
        position: 'relative',
        pointerEvents: 'auto',
      }}
      onError={() => setHasError(true)}
    >
      <Spline
        scene={scene}
        onError={() => setHasError(true)}
      />
    </Box>
  );
}
