import React from 'react';

const ScanLine: React.FC = () => {
  return (
    <div
      className="scan-line"
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '2px',
        background: 'linear-gradient(90deg, transparent, hsl(200, 100%, 55%), transparent)',
        animation: 'scanLine 3s linear infinite',
        pointerEvents: 'none',
        zIndex: 9999,
      }}
    />
  );
};

export default ScanLine;
