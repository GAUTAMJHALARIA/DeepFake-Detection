'use client'

import React, { Suspense, lazy } from 'react'
import { Box } from '@mui/material'
const Spline = lazy(() => import('@splinetool/react-spline'))

interface SplineSceneProps {
  scene: string
  className?: string
}

export function SplineScene({ scene, className }: SplineSceneProps) {
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
          }}
        >
          <span className="loader"></span>
        </Box>
      }
    >
      <Box
        sx={{
          width: '100%',
          height: '100%',
          minWidth: '100%',
          minHeight: '100%',
          position: 'relative',
          pointerEvents: 'auto',
        }}
      >
        <Spline
          scene={scene}
        />
      </Box>
    </Suspense>
  );
}
