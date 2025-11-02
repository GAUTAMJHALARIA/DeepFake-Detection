import React from 'react';
import { Box } from '@mui/material';
import DynamicHero from '../components/DynamicHero';
import CreativeFeatures from '../components/CreativeFeatures';
import { useNavigate } from 'react-router-dom';

const HomePage: React.FC = () => {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate('/analyze');
  };

  return (
    <Box>
      <DynamicHero onGetStarted={handleGetStarted} />
      <CreativeFeatures />
    </Box>
  );
};

export default HomePage;
