import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Box, Typography, Button, Paper } from '@mui/material';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <Paper sx={{ p: 3, m: 2, textAlign: 'center' }}>
          <Typography variant="h6" color="error" gutterBottom>
            Something went wrong with the video player
          </Typography>
          <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
            This is likely due to a hot reload issue or unsupported video codec.
          </Typography>
          <Button
            variant="contained"
            onClick={() => this.setState({ hasError: false })}
          >
            Try Again
          </Button>
          {this.state.error && (
            <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.100', borderRadius: 1 }}>
              <Typography variant="caption" component="pre">
                {this.state.error.message}
              </Typography>
            </Box>
          )}
        </Paper>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
