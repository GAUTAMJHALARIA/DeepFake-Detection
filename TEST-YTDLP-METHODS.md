# Testing yt-dlp URL Downloads

## 🎯 Overview
The system supports downloading and analyzing videos directly from URLs using `yt-dlp`.

---

## 📋 Supported Platforms

The system can download from:
- ✅ **YouTube** - Videos, shorts, live streams
- ✅ **Twitter/X** - Video posts
- ✅ **Vimeo** - Videos
- ✅ **Instagram** - Video posts
- ✅ **TikTok** - Videos
- ✅ **Facebook** - Videos
- ✅ **Any platform** supported by yt-dlp

---

## 🔧 Testing Methods

### **Method 1: Using Python Test Script**

```bash
# Basic usage
python test-ytdlp-download.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Example
python test-ytdlp-download.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

The script will:
1. Send URL to API
2. Download video with yt-dlp
3. Process and analyze
4. Display results
5. Save JSON output

### **Method 2: Using cURL**

```bash
curl -X POST "http://localhost:8000/predict-url" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer change-me" \
  -d '{
    "url": "https://www.youtube.com/watch?v=VIDEO_ID"
  }'
```

### **Method 3: Using Python Requests**

```python
import requests

url = "http://localhost:8000/predict-url"
headers = {
    "Authorization": "Bearer change-me",
    "Content-Type": "application/json"
}
data = {
    "url": "https://www.youtube.com/watch?v=VIDEO_ID"
}

response = requests.post(url, json=data, headers=headers)
result = response.json()
print(result)
```

### **Method 4: From Frontend (Next Step)**

Currently, the frontend doesn't have URL upload UI, but you can add it:

```typescript
const analyzeURL = async (url: string) => {
  const response = await axios.post(
    'http://localhost:8000/predict-url',
    { url },
    { headers: { Authorization: 'Bearer change-me' } }
  );
  setResult(response.data);
};
```

---

## 🧪 Example Test Cases

### **YouTube Video**
```bash
python test-ytdlp-download.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### **YouTube Short**
```bash
python test-ytdlp-download.py "https://www.youtube.com/shorts/VIDEO_ID"
```

### **Twitter/X Video**
```bash
python test-ytdlp-download.py "https://twitter.com/user/status/TWEET_ID"
```

### **Vimeo Video**
```bash
python test-ytdlp-download.py "https://vimeo.com/VIDEO_ID"
```

### **Direct Video URL**
```bash
python test-ytdlp-download.py "https://example.com/video.mp4"
```

---

## 🔍 How It Works

### **Backend Implementation**

1. **Receive URL**: `POST /predict-url` endpoint
2. **Download Video**: Use `yt-dlp` to extract video URL
3. **Fetch Content**: Download video bytes
4. **Process**: Same pipeline as file upload
5. **Return**: Analysis results

### **Code Flow**

```python
# 1. Download video from URL
def download_video_from_url(url: str) -> bytes:
    ydl_opts = {
        "format": "best[height<=1080]",  # Limit to 1080p
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_url = info["url"]

        response = requests.get(video_url, timeout=60)
        return response.content

# 2. Process like regular video
video_content = download_video_from_url(url)
result = analyze_video_enhanced(video_content, analysis_id)
```

---

## ⚙️ Configuration

### **Video Quality**
```python
# api/app/enhanced_inference.py
ydl_opts = {
    "format": "best[height<=1080]",  # Max 1080p
    "quiet": True,
    "no_warnings": True,
}
```

### **Timeout Settings**
```python
REQUEST_TIMEOUT = 30.0  # API timeout
video_timeout = 60     # Video download timeout
```

---

## 🐛 Troubleshooting

### **Error: "Failed to download video from URL"**

**Solutions:**
1. Check internet connection
2. Verify URL is accessible
3. Try with a different video URL
4. Check yt-dlp installation: `pip install yt-dlp`

### **Error: "Video format not supported"**

**Solutions:**
1. yt-dlp will automatically select best format
2. Check if video is age-restricted
3. Try downloading directly from the platform

### **Error: "Video too long"**

**Solutions:**
1. Limit video duration in settings
2. Use `MAX_FRAMES` setting
3. Sample at lower FPS: `fps=1.0`

### **Check Logs**
```bash
# View API logs
docker logs dfd-api-1

# Or if running locally
tail -f api/logs/server.log
```

---

## 📊 Expected Response

```json
{
  "id": "analysis-uuid",
  "score": 0.75,
  "label": "fake",
  "source_url": "https://www.youtube.com/watch?v=...",
  "video_info": {
    "duration": 30.5,
    "fps": 29.97,
    "resolution": "1920x1080",
    "processed_frames": 914
  },
  "frames": [...],
  "statistics": {...}
}
```

---

## 🚀 Quick Start

1. **Start Services**
```bash
docker compose up -d
```

2. **Test URL Download**
```bash
python test-ytdlp-download.py "YOUTUBE_URL"
```

3. **View Results**
- JSON file saved as `result_[uuid].json`
- Can also access via API: `GET /analysis/{id}`

---

## 📝 Notes

- **Quality**: Downloads best available quality up to 1080p
- **Format**: Automatically converts to browser-compatible format
- **Caching**: Results cached in Redis for 30 minutes
- **Privacy**: URLs are logged for debugging purposes
- **Error Handling**: Graceful fallback if download fails

---

## 🔗 Related Endpoints

```http
POST /predict-url              # URL analysis
GET /analysis/{id}             # Get cached results
GET /frames/{id}/{index}        # Get specific frame
GET /thumbnails/{id}           # Get thumbnails
GET /gradcam/{id}/{index}      # Get Grad-CAM
```

---

**Happy Testing! 🎬**
