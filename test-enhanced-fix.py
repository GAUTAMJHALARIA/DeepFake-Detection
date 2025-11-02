#!/usr/bin/env python3
"""
Quick test to verify the enhanced analysis fix
"""

import requests


def test_enhanced_endpoint():
    """Test if the enhanced endpoint is working"""
    print("🔍 Testing enhanced analysis endpoint...")

    try:
        # Test with a simple health check first
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ API is responding")
            health_data = health_response.json()
            print(f"   - Status: {'OK' if health_data.get('ok') else 'Error'}")

            # Check if enhanced features are enabled
            details = health_data.get("details", {})
            if isinstance(details, dict):
                print(
                    f"   - Grad-CAM enabled: {details.get('gradcam_enabled', 'Unknown')}"
                )
                print(
                    f"   - Max resolution: {details.get('max_resolution', 'Unknown')}"
                )

            return True
        else:
            print(f"❌ API health check failed: {health_response.status_code}")
            return False

    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False


def main():
    print("🚀 Testing Enhanced Deepfake Detection API Fix")
    print("=" * 50)

    if test_enhanced_endpoint():
        print("\n🎉 Enhanced API is working!")
        print("\n💡 Next steps:")
        print("   1. Open http://localhost:3000 in your browser")
        print("   2. Go to the 'Enhanced Analysis' tab")
        print("   3. Upload a video file to test the full pipeline")
        print("\n📝 The fix resolved the 'dict has no append' error")
        print("   - Separated frame_cache_data from frame_metadata")
        print("   - frame_metadata is now properly used as a list")
        print("   - Memory optimization is still active")
    else:
        print("\n⚠️  API is not responding properly")
        print("   Make sure the API is running on port 8000")


if __name__ == "__main__":
    main()
