import React from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import {
  CssBaseline,
  Box,
  AppBar,
  Toolbar,
  Typography,
  Container,
  Chip
} from '@mui/material';
import {
  Security,
  Analytics
} from '@mui/icons-material';

import EnhancedVideoAnalysis from './components/EnhancedVideoAnalysis';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#0a0a0a',
      paper: '#1a1a1a',
    },
  },
  typography: {
    h4: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 500,
    },
  },
});

function App() {

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1, minHeight: '100vh' }}>
        <AppBar position="static" sx={{ background: 'linear-gradient(45deg, #1976d2 30%, #21CBF3 90%)' }}>
          <Toolbar>
            <Security sx={{ mr: 2 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              Deepfake Detection System
            </Typography>
            <Box display="flex" alignItems="center" gap={2}>
              <Analytics sx={{ mr: 1 }} />
              <Chip
                label="AI-Powered Analysis"
                color="secondary"
                size="small"
                sx={{ fontWeight: 'bold' }}
              />
              <Typography variant="body2" sx={{ opacity: 0.8 }}>
                Frame-by-Frame Deepfake Detection
              </Typography>
            </Box>
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 2 }}>
          <EnhancedVideoAnalysis />
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
