import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';

import Navigation from './components/Navigation';
import Footer from './components/Footer';
import HomePage from './pages/HomePage';
import AnalyzePage from './pages/AnalyzePage';
import FeaturesPage from './pages/FeaturesPage';
import AboutPage from './pages/AboutPage';
import ParticleBackground from './components/ui/ParticleBackground';
import ScanLine from './components/ui/ScanLine';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00AAFF', // Electric Blue
      light: '#33CCFF',
      dark: '#0088CC',
    },
    secondary: {
      main: '#FF00FF', // Magenta
      light: '#FF33FF',
      dark: '#CC00CC',
    },
    background: {
      default: 'hsl(0, 0%, 0%)', // Deep black
      paper: 'hsl(0, 0%, 3%)',
    },
    text: {
      primary: 'hsl(0, 0%, 95%)',
      secondary: 'hsl(220, 15%, 70%)',
    },
    divider: 'hsla(200, 100%, 55%, 0.2)',
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h4: {
      fontWeight: 700,
      letterSpacing: '-0.02em',
    },
    h6: {
      fontWeight: 600,
      letterSpacing: '0.01em',
    },
  },
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: 'hsla(0, 0%, 2%, 0.85)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid hsla(200, 100%, 55%, 0.2)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          backgroundColor: 'hsla(0, 0%, 3%, 0.8)',
          backdropFilter: 'blur(20px)',
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: 'hsla(0, 0%, 3%, 0.8)',
          backdropFilter: 'blur(20px)',
          border: '1px solid hsla(200, 100%, 55%, 0.2)',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
          '&:hover': {
            borderColor: 'hsl(200, 100%, 55%)',
            boxShadow: '0 0 20px hsla(200, 100%, 55%, 0.3), 0 0 40px hsla(200, 100%, 55%, 0.1)',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 600,
          borderRadius: '8px',
        },
        contained: {
          background: 'linear-gradient(135deg, hsl(200, 100%, 55%) 0%, hsl(210, 100%, 60%) 100%)',
          color: 'hsl(0, 0%, 0%)',
          boxShadow: '0 0 20px hsla(200, 100%, 55%, 0.4)',
          '&:hover': {
            background: 'linear-gradient(135deg, hsl(200, 100%, 60%) 0%, hsl(210, 100%, 65%) 100%)',
            boxShadow: '0 0 30px hsla(200, 100%, 55%, 0.6)',
          },
        },
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          backgroundColor: 'hsla(200, 100%, 55%, 0.15)',
          border: '1px solid hsla(200, 100%, 55%, 0.3)',
          color: 'hsl(200, 100%, 55%)',
        },
      },
    },
  },
});

function App() {
  return (
    <Router>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Box
          sx={{
            flexGrow: 1,
            minHeight: '100vh',
            position: 'relative',
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column',
          }}
          className="animated-gradient"
        >
          {/* Sci-Fi Effects */}
          <ParticleBackground />
          <ScanLine />

          {/* Animated Grid Lines */}
          <Box
            sx={{
              position: 'fixed',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              pointerEvents: 'none',
              zIndex: 0,
              opacity: 0.15,
              backgroundImage: `
                linear-gradient(hsla(0, 0%, 100%, 0.05) 1px, transparent 1px),
                linear-gradient(90deg, hsla(0, 0%, 100%, 0.05) 1px, transparent 1px)
              `,
              backgroundSize: '50px 50px',
              animation: 'gridMove 20s linear infinite',
            }}
          />

          {/* Navigation */}
          <Navigation />

          {/* Routes */}
          <Box sx={{ flex: 1 }}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/analyze" element={<AnalyzePage />} />
              <Route path="/features" element={<FeaturesPage />} />
              <Route path="/about" element={<AboutPage />} />
            </Routes>
          </Box>

          {/* Footer */}
          <Footer />
        </Box>
      </ThemeProvider>
    </Router>
  );
}

export default App;
