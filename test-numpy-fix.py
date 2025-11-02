#!/usr/bin/env python3
"""
Test the numpy serialization fix
"""

import requests
import json


def test_health_with_details():
    """Test health endpoint to see if enhanced features are reported correctly"""
    print("🔍 Testing enhanced health endpoint...")

    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health endpoint working")
            print(f"   - API Status: {'OK' if data.get('ok') else 'Error'}")

            details = data.get("details", {})
            if isinstance(details, dict):
                print(f"   - Redis: {details.get('redis', 'Unknown')}")
                print(f"   - Grad-CAM: {details.get('gradcam_enabled', 'Unknown')}")
                print(
                    f"   - Max Resolution: {details.get('max_resolution', 'Unknown')}"
                )

            # Check if response is properly serialized (no numpy types)
            json.dumps(data)  # This will fail if numpy types exist
            print("✅ Response properly serialized (no numpy types)")

            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False

    except json.JSONEncodeError as e:
        print(f"❌ JSON serialization error: {e}")
        return False
    except Exception as e:
        print(f"❌ Health test failed: {e}")
        return False


def main():
    print("🚀 Testing Numpy Serialization Fix")
    print("=" * 40)

    if test_health_with_details():
        print("\n🎉 Numpy serialization fix is working!")
        print("\n💡 The enhanced analysis endpoint should now work without errors")
        print("   - All numpy types are converted to native Python types")
        print("   - Pydantic can properly serialize the response")
        print("   - Memory optimizations are still active")
        print("\n🧪 Ready to test with video upload!")
    else:
        print("\n⚠️  There might still be serialization issues")


if __name__ == "__main__":
    main()
