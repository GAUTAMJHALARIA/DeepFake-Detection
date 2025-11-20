import { useEffect, useRef } from 'react';

/**
 * Custom hook for debounced canvas resize with requestAnimationFrame
 * Prevents forced reflows by batching resize operations
 */
export function useDebouncedResize(
  callback: (width: number, height: number) => void,
  debounceMs: number = 150
) {
  const rafIdRef = useRef<number | null>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const handleResize = () => {
      // Clear any pending timeout
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      // Debounce the resize
      timeoutRef.current = setTimeout(() => {
        // Cancel any pending animation frame
        if (rafIdRef.current !== null) {
          cancelAnimationFrame(rafIdRef.current);
        }

        // Use requestAnimationFrame to batch resize operations
        rafIdRef.current = requestAnimationFrame(() => {
          callback(window.innerWidth, window.innerHeight);
          rafIdRef.current = null;
        });
      }, debounceMs);
    };

    // Initial call
    callback(window.innerWidth, window.innerHeight);

    window.addEventListener('resize', handleResize, { passive: true });

    return () => {
      window.removeEventListener('resize', handleResize);
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
      }
    };
  }, [callback, debounceMs]);
}
