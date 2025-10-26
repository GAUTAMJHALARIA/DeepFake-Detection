import React, { useRef, useState } from 'react';
import { Box, Typography, Card, CardContent, Paper, Tooltip } from '@mui/material';
import { motion } from 'framer-motion';

interface Frame {
  index: number;
  timestamp: number;
  confidence: number;
  label: string;
}

interface AnimatedConfidenceTimelineProps {
  frames: Frame[];
  currentFrameIndex: number;
  onFrameSelect: (index: number) => void;
}

const AnimatedConfidenceTimeline: React.FC<AnimatedConfidenceTimelineProps> = ({
  frames,
  currentFrameIndex,
  onFrameSelect,
}) => {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [hoverPosition, setHoverPosition] = useState<{ x: number; y: number } | null>(null);

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.7) return '#f44336'; // Red
    if (confidence >= 0.3) return '#ff9800'; // Orange
    return '#4caf50'; // Green
  };

  const handleMouseMove = (e: React.MouseEvent, index: number) => {
    setHoveredIndex(index);
    setHoverPosition({ x: e.clientX, y: e.clientY });
  };

  const handleMouseLeave = () => {
    setHoveredIndex(null);
    setHoverPosition(null);
  };

  const handleClick = (index: number) => {
    onFrameSelect(index);
  };

  return (
    <Card sx={{ mb: 3, background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)' }}>
      <CardContent>
        <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold' }}>
          📊 Animated Confidence Timeline
        </Typography>
        <Typography variant="body2" color="textSecondary" paragraph>
          Interactive visualization of deepfake confidence across entire video
        </Typography>

        {/* Timeline Container */}
        <Paper
          sx={{
            p: 2,
            minHeight: 180,
            position: 'relative',
            background: 'white',
            overflow: 'hidden',
            borderRadius: 2,
          }}
          onMouseLeave={handleMouseLeave}
        >
          {/* Confidence Bars */}
          <Box
            sx={{
              display: 'flex',
              gap: '2px',
              height: 120,
              alignItems: 'flex-end',
              position: 'relative'
            }}
          >
            {frames.map((frame, index) => {
              const isSelected = index === currentFrameIndex;
              const isHovered = hoveredIndex === index;

              return (
                <motion.div
                  key={frame.index}
                  initial={{ opacity: 0, scaleY: 0 }}
                  animate={{
                    opacity: 1,
                    scaleY: 1,
                    transition: { delay: index * 0.002, duration: 0.5 }
                  }}
                  whileHover={{ scaleY: 1.15, originY: 1 }}
                  onMouseMove={(e) => handleMouseMove(e as any, index)}
                  onClick={() => handleClick(index)}
                  style={{
                    flex: 1,
                    backgroundColor: getConfidenceColor(frame.confidence),
                    minWidth: '2px',
                    height: `${Math.max(frame.confidence * 100, 5)}%`,
                    cursor: 'pointer',
                    borderRadius: '2px 2px 0 0',
                    borderTop: isSelected ? '4px solid #2196F3' : 'none',
                    borderLeft: isSelected ? '2px solid #2196F3' : 'none',
                    borderRight: isSelected ? '2px solid #2196F3' : 'none',
                    filter: isHovered ? 'brightness(1.3) saturate(1.2)' : 'brightness(1)',
                    transition: 'all 0.2s ease',
                    boxShadow: isHovered ? '0 4px 8px rgba(0,0,0,0.2)' : 'none',
                    zIndex: isHovered ? 10 : 1,
                  }}
                />
              );
            })}
          </Box>

          {/* Wave Effect Overlay */}
          <Box
            sx={{
              position: 'absolute',
              bottom: 0,
              left: 0,
              right: 0,
              height: '30%',
              background: 'linear-gradient(to top, rgba(33, 150, 243, 0.1), transparent)',
              pointerEvents: 'none',
            }}
          />

          {/* Time Labels */}
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1, px: 1 }}>
            <Typography variant="caption" color="textSecondary" fontWeight="600">
              0s
            </Typography>
            <Typography variant="caption" color="textSecondary" fontWeight="600">
              {frames.length > 0 ? frames[frames.length - 1].timestamp.toFixed(1) + 's' : '0s'}
            </Typography>
          </Box>
        </Paper>

        {/* Hover Tooltip */}
        {hoveredIndex !== null && hoverPosition && (
          <Tooltip
            open={true}
            title={
              <Box>
                <Typography variant="body2" fontWeight="bold">
                  Frame {frames[hoveredIndex].index + 1}
                </Typography>
                <Typography variant="body2">
                  Confidence: {(frames[hoveredIndex].confidence * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2">
                  Label: {frames[hoveredIndex].label.toUpperCase()}
                </Typography>
                <Typography variant="body2">
                  Time: {frames[hoveredIndex].timestamp.toFixed(2)}s
                </Typography>
              </Box>
            }
            arrow
          >
            <Box
              sx={{
                position: 'fixed',
                left: hoverPosition.x,
                top: hoverPosition.y - 60,
                pointerEvents: 'none',
                zIndex: 9999,
              }}
            />
          </Tooltip>
        )}

        {/* Legend */}
        <Box sx={{ display: 'flex', gap: 3, mt: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box
              width={20}
              height={20}
              bgcolor="#4caf50"
              borderRadius="50%"
              sx={{ boxShadow: '0 2px 4px rgba(0,0,0,0.2)' }}
            />
            <Typography variant="caption" fontWeight="600">Real (&lt;30%)</Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box
              width={20}
              height={20}
              bgcolor="#ff9800"
              borderRadius="50%"
              sx={{ boxShadow: '0 2px 4px rgba(0,0,0,0.2)' }}
            />
            <Typography variant="caption" fontWeight="600">Uncertain (30-70%)</Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box
              width={20}
              height={20}
              bgcolor="#f44336"
              borderRadius="50%"
              sx={{ boxShadow: '0 2px 4px rgba(0,0,0,0.2)' }}
            />
            <Typography variant="caption" fontWeight="600">Fake (&gt;70%)</Typography>
          </Box>
        </Box>

        {/* Key Statistics */}
        <Box sx={{ mt: 2, display: 'flex', gap: 2, justifyContent: 'center', flexWrap: 'wrap' }}>
          <Paper sx={{ px: 2, py: 1 }}>
            <Typography variant="caption" color="textSecondary">Total Frames:</Typography>
            <Typography variant="h6" fontWeight="bold" color="primary">
              {frames.length}
            </Typography>
          </Paper>
          <Paper sx={{ px: 2, py: 1 }}>
            <Typography variant="caption" color="textSecondary">Avg Confidence:</Typography>
            <Typography variant="h6" fontWeight="bold" color="error">
              {(frames.reduce((sum, f) => sum + f.confidence, 0) / frames.length * 100).toFixed(1)}%
            </Typography>
          </Paper>
          <Paper sx={{ px: 2, py: 1 }}>
            <Typography variant="caption" color="textSecondary">Suspicious Frames:</Typography>
            <Typography variant="h6" fontWeight="bold" color="warning.main">
              {frames.filter(f => f.confidence >= 0.7).length}
            </Typography>
          </Paper>
        </Box>
      </CardContent>
    </Card>
  );
};

export default AnimatedConfidenceTimeline;
