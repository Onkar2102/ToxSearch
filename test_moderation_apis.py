#!/usr/bin/env python3
"""
Test script to verify both Google Perspective API and OpenAI moderation are working.
Run this to ensure your API keys are properly configured.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_google_moderation():
    """Test Google Perspective API"""
    print("🔍 Testing Google Perspective API...")
    
    try:
        from gne.google_moderation import evaluate_moderation
        
        # Test with a simple text
        test_text = "This is a test message to verify the API is working."
        result = evaluate_moderation(test_text)
        
        if result:
            print("✅ Google Perspective API: SUCCESS")
            print(f"   Response ID: {result.get('id', 'N/A')}")
            print(f"   Model: {result.get('model', 'N/A')}")
            print(f"   Results count: {len(result.get('results', []))}")
            return True
        else:
            print("❌ Google Perspective API: FAILED - No result returned")
            return False
            
    except Exception as e:
        print(f"❌ Google Perspective API: ERROR - {e}")
        return False

def test_openai_moderation():
    """Test OpenAI Moderation API"""
    print("🔍 Testing OpenAI Moderation API...")
    
    try:
        from gne.openai_moderation import evaluate_moderation
        
        # Test with a simple text
        test_text = "This is a test message to verify the API is working."
        result = evaluate_moderation(test_text)
        
        if result:
            print("✅ OpenAI Moderation API: SUCCESS")
            print(f"   Response ID: {result.get('id', 'N/A')}")
            print(f"   Model: {result.get('model', 'N/A')}")
            print(f"   Results count: {len(result.get('results', []))}")
            return True
        else:
            print("❌ OpenAI Moderation API: FAILED - No result returned")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI Moderation API: ERROR - {e}")
        return False

def test_environment_variables():
    """Check if required environment variables are set"""
    print("🔍 Checking environment variables...")
    
    google_key = os.getenv("PERSPECTIVE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if google_key:
        print("✅ PERSPECTIVE_API_KEY: SET")
    else:
        print("❌ PERSPECTIVE_API_KEY: NOT SET")
    
    if openai_key:
        print("✅ OPENAI_API_KEY: SET")
    else:
        print("❌ OPENAI_API_KEY: NOT SET")
    
    return bool(google_key and openai_key)

def main():
    """Run all tests"""
    print("🚀 Testing Moderation APIs...")
    print("=" * 50)
    
    # Check environment variables
    env_ok = test_environment_variables()
    print()
    
    if not env_ok:
        print("⚠️  Some environment variables are missing. Please check your .env file.")
        return
    
    # Test Google API
    google_ok = test_google_moderation()
    print()
    
    # Test OpenAI API
    openai_ok = test_openai_moderation()
    print()
    
    # Summary
    print("=" * 50)
    print("📊 TEST SUMMARY:")
    print(f"   Environment Variables: {'✅' if env_ok else '❌'}")
    print(f"   Google Perspective API: {'✅' if google_ok else '❌'}")
    print(f"   OpenAI Moderation API: {'✅' if openai_ok else '❌'}")
    
    if google_ok and openai_ok:
        print("\n🎉 Both APIs are working! You can choose which one to use.")
        print("   - Google: Better for research, no rate limits")
        print("   - OpenAI: Higher accuracy, more categories")
    elif google_ok:
        print("\n✅ Google API is working. Consider using it for your evolution system.")
    elif openai_ok:
        print("\n✅ OpenAI API is working. Consider using it for your evolution system.")
    else:
        print("\n❌ Both APIs failed. Please check your configuration.")

if __name__ == "__main__":
    main()
