/**
 * API Configuration
 *
 * Centralized API URL configuration for the application.
 * Uses REACT_APP_API_URL environment variable, falls back to localhost for development.
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// API Token (if authentication is enabled)
const API_TOKEN = process.env.REACT_APP_API_TOKEN || 'change-me';

/**
 * Get the full API URL for an endpoint
 */
export const getApiUrl = (endpoint: string): string => {
  // Remove leading slash if present to avoid double slashes
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint.slice(1) : endpoint;
  return `${API_BASE_URL}/${cleanEndpoint}`;
};

/**
 * Get default headers for API requests
 * @param additionalHeaders - Additional headers to merge
 * @param skipContentType - Skip setting Content-Type (useful for FormData)
 */
export const getApiHeaders = (
  additionalHeaders: Record<string, string> = {},
  skipContentType: boolean = false
): Record<string, string> => {
  const headers: Record<string, string> = {};

  // Only set Content-Type if not skipped (for FormData, axios will set it automatically)
  if (!skipContentType && !additionalHeaders['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }

  // Merge additional headers (they take precedence)
  Object.assign(headers, additionalHeaders);

  // Add authorization header if token is set and not the default
  if (API_TOKEN && API_TOKEN !== 'change-me') {
    headers['Authorization'] = `Bearer ${API_TOKEN}`;
  }

  return headers;
};

/**
 * API Endpoints
 */
export const API_ENDPOINTS = {
  PREDICT: getApiUrl('predict'),
  PREDICT_URL: getApiUrl('predict-url'),
  HEALTH: getApiUrl('health'),
  ANALYSIS: (id: string) => getApiUrl(`analysis/${id}`),
  FRAME: (analysisId: string, frameIndex: number) => getApiUrl(`frames/${analysisId}/${frameIndex}`),
  GRADCAM: (analysisId: string, frameIndex: number) => getApiUrl(`gradcam/${analysisId}/${frameIndex}`),
  THUMBNAILS: (analysisId: string) => getApiUrl(`thumbnails/${analysisId}`),
  VIDEO: (analysisId: string) => getApiUrl(`video/${analysisId}`),
  SUPPORTED_FORMATS: getApiUrl('supported-formats'),
};

export default {
  BASE_URL: API_BASE_URL,
  TOKEN: API_TOKEN,
  getApiUrl,
  getApiHeaders,
  ENDPOINTS: API_ENDPOINTS,
};
