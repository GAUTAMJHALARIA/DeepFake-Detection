#!/usr/bin/env python3
"""
Test yt-dlp URL download feature
Usage: python test-ytdlp-download.py [URL]
"""

import sys
import requests
import json

# Configuration
API_URL = "http://localhost:8000"
ENDPOINT = "/predict-url"
API_KEY = "change-me"  # JWT Secret if REQUIRE_AUTH=true


def test_url_download(url: str):
    """Test URL download and analysis"""
    print(f"\n{'='*60}")
    print("Testing yt-dlp URL Download")
    print(f"{'='*60}\n")
    print(f"📍 URL: {url}")
    print(f"🔗 API Endpoint: {API_URL}{ENDPOINT}")
    print("\n⏳ Processing...\n")

    try:
        # Prepare request
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {"url": url}

        # Send request
        response = requests.post(
            f"{API_URL}{ENDPOINT}",
            json=payload,
            headers=headers,
            timeout=300,  # 5 minutes for long videos
        )

        # Check response
        if response.status_code == 200:
            result = response.json()

            print("✅ Analysis Complete!\n")
            print("📊 Results:")
            print(f"   ID: {result.get('id')}")
            print(f"   Score: {result.get('score'):.3f}")
            print(f"   Label: {result.get('label').upper()}")
            print(f"   Latency: {result.get('latency_ms')} ms")
            print(f"   Source URL: {result.get('source_url')}")

            # Video info
            video_info = result.get("video_info", {})
            print("\n🎬 Video Info:")
            print(f"   Duration: {video_info.get('duration', 0):.2f}s")
            print(f"   FPS: {video_info.get('fps', 0):.2f}")
            print(f"   Resolution: {video_info.get('resolution', 'N/A')}")
            print(f"   Frames: {video_info.get('processed_frames', 0)}")

            # Statistics
            stats = result.get("statistics", {})
            print("\n📈 Statistics:")
            print(f"   Mean Confidence: {stats.get('mean_confidence', 0):.3f}")
            print(f"   Max Confidence: {stats.get('max_confidence', 0):.3f}")
            print(f"   Min Confidence: {stats.get('min_confidence', 0):.3f}")
            print(f"   Suspicious Frames: {stats.get('suspicious_frames', 0)}")

            # Save result
            output_file = f"result_{result.get('id')}.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {output_file}")

            return result

        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   Message: {response.text}")
            return None

    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Cannot connect to API at {API_URL}")
        print("   Make sure the API server is running!")
        return None
    except requests.exceptions.Timeout:
        print("❌ Error: Request timed out")
        print("   Video might be too long or network is slow")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def main():
    """Main function"""

    # Check for URL argument
    if len(sys.argv) < 2:
        print("Usage: python test-ytdlp-download.py <URL>")
        print("\nExample URLs:")
        print("  - YouTube: https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        print("  - Twitter: https://twitter.com/user/status/123456")
        print("  - Vimeo: https://vimeo.com/123456")
        print("\nTest with a known deepfake video URL")
        sys.exit(1)

    url = sys.argv[1]

    # Test the URL download
    result = test_url_download(url)

    if result:
        print(f"\n{'='*60}")
        print("✅ Test completed successfully!")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print("❌ Test failed!")
        print(f"{'='*60}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
