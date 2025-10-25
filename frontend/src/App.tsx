import React, { useState } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import {
  CssBaseline,
  Box,
  AppBar,
  Toolbar,
  Typography,
  Container,
  Tabs,
  Tab,
  Chip
} from '@mui/material';
import {
  Security,
  VideoLibrary,
  Analytics
} from '@mui/icons-material';

import VideoUpload from './components/VideoUpload';
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

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

function App() {
  const [tabValue, setTabValue] = useState(0);

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

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
              <Chip
                label="Enhanced Edition"
                color="secondary"
                size="small"
                sx={{ fontWeight: 'bold' }}
              />
              <Typography variant="body2" sx={{ opacity: 0.8 }}>
                Advanced AI-Powered Video Analysis
              </Typography>
            </Box>
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 2 }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
            <Tabs
              value={tabValue}
              onChange={handleTabChange}
              aria-label="analysis tabs"
              variant="scrollable"
              scrollButtons="auto"
            >
              <Tab
                icon={<VideoLibrary />}
                label="Basic Analysis"
                id="tab-0"
                aria-controls="tabpanel-0"
              />
              <Tab
                icon={<Analytics />}
                label="Enhanced Analysis"
                id="tab-1"
                aria-controls="tabpanel-1"
              />
            </Tabs>
          </Box>

          <TabPanel value={tabValue} index={0}>
            <VideoUpload />
          </TabPanel>

          <TabPanel value={tabValue} index={1}>
            <EnhancedVideoAnalysis />
          </TabPanel>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
