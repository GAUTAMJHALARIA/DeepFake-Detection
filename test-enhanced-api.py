#!/usr/bin/env python3
"""
Test script for Enhanced Deepfake Detection API
"""

import requests
import time

API_BASE = "http://localhost:8000"
AUTH_HEADER = {"Authorization": "Bearer change-me"}


def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   - API Status: {'OK' if data.get('ok') else 'Error'}")
            if isinstance(data.get("details"), dict):
                print(f"   - Redis: {data['details'].get('redis', 'Unknown')}")
                print(
                    f"   - Grad-CAM: {data['details'].get('gradcam_enabled', 'Unknown')}"
                )
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_supported_formats():
    """Test supported formats endpoint"""
    print("\n🔍 Testing supported formats...")
    try:
        response = requests.get(f"{API_BASE}/supported-formats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Supported formats retrieved")
            print(f"   - Video formats: {len(data.get('video_formats', []))}")
            print(f"   - Image formats: {len(data.get('image_formats', []))}")
            return True
        else:
            print(f"❌ Supported formats failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Supported formats error: {e}")
        return False


def test_redis_connection():
    """Test Redis connection indirectly"""
    print("\n🔍 Testing Redis connection...")
    try:
        # Try to access an endpoint that uses Redis
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            redis_status = data.get("details", {}).get("redis", "unknown")
            if redis_status == "connected":
                print("✅ Redis connection working")
                return True
            else:
                print(f"❌ Redis connection issue: {redis_status}")
                return False
        else:
            print("❌ Cannot test Redis connection")
            return False
    except Exception as e:
        print(f"❌ Redis test error: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 Testing Enhanced Deepfake Detection API")
    print("=" * 50)

    tests = [
        test_health,
        test_supported_formats,
        test_redis_connection,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        time.sleep(0.5)

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed! Enhanced API is ready.")
        print("\n💡 Next steps:")
        print("   1. Start the frontend: cd frontend && npm start")
        print("   2. Open http://localhost:3000")
        print("   3. Try the Enhanced Analysis tab")
    else:
        print("⚠️  Some tests failed. Check the API configuration.")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure TensorFlow Serving is running on port 8501")
        print("   2. Ensure Redis is running on port 6379")
        print("   3. Check API logs for detailed errors")


if __name__ == "__main__":
    main()
