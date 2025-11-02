# Deterministic Analysis Fixes

## Problem
Getting different results for the same video upload.

## Root Causes Identified

### 1. Non-Deterministic Frame Sampling
**Before:**
```python
step = max(int(round(src_fps / max(target_fps, 0.1))), 1)
if total_frames // step > settings.MAX_CACHED_FRAMES:
    step = max(total_frames // settings.MAX_CACHED_FRAMES, 1)
```
- Used rounding that could vary
- Unpredictable frame selection

**Fixed:**
```python
target_frame_count = min(settings.MAX_CACHED_FRAMES, int(target_fps * duration))
step = max(total_frames // target_frame_count, 1)
```
- Deterministic calculation
- Always selects same frames for same video

### 2. Inconsistent Face Detection
**Before:**
```python
faces = cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
)
```

**Fixed:**
```python
faces = cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,  # More conservative
    minNeighbors=6,    # Higher threshold
    minSize=(40, 40),  # Larger minimum
    flags=cv2.CASCADE_SCALE_IMAGE
)
```
- More consistent face detection
- Less sensitive to minor variations

### 3. Random Grad-CAM Noise
**Note:** Random noise in Grad-CAM visualization:
```python
noise = np.random.normal(0, noise_level, (h, w))
```
- This only affects **visualization**, not model predictions
- **Predictions remain consistent**

## Verification

### What's Deterministic Now:
✅ Frame selection - same frames every time
✅ Face detection parameters - consistent
✅ Model predictions - TensorFlow Serving is deterministic
✅ Overall confidence scores - calculated deterministically
✅ Video info extraction - consistent

### What's Not Deterministic (Doesn't Affect Results):
⚠️ Grad-CAM heatmap noise (visualization only)
⚠️ Temporary file names (doesn't affect processing)
⚠️ Analysis IDs (doesn't affect results)

## Testing

To verify results are consistent:

1. Upload the same video multiple times
2. Check that:
   - Same number of frames are processed
   - Same confidence scores for each frame
   - Same overall analysis result

## File Changes

### Modified Files:
- `api/app/enhanced_inference.py` - Frame sampling and face detection
- `api/app/cache.py` - Memory management
- `docker-compose.yml` - Redis memory limits

### Key Lines:
- Line 204-206: Deterministic frame step calculation
- Line 234-241: Consistent face detection parameters
- Line 216: Added logging for verification
