'use client'

import React, { Suspense, lazy } from 'react';
import { Box } from '@mui/material';

// Lazy load SplineScene for performance
const SplineScene = lazy(() => import('./splite').then(module => ({ default: module.SplineScene })));

export function SplineSceneBasic() {
  return (
    <Suspense
      fallback={
        <Box
          sx={{
            width: '100%',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            background: 'transparent',
          }}
        >
          <span className="loader"></span>
        </Box>
      }
    >
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          position: 'relative',
          background: 'transparent',
          overflow: 'visible',
          minWidth: '100%',
          minHeight: '100%',
          pointerEvents: 'auto',
        }}
      >
        <SplineScene
          scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode"
        />
      </div>
    </Suspense>
  );
}
