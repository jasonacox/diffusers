#!/usr/bin/env python3
"""
Test script for the updated serverasync.py with Flux support
"""

import requests
import json
import sys

def test_flux_model():
    """Test Flux model generation"""
    url = "http://localhost:8500/api/diffusers/inference"
    
    payload = {
        "prompt": "A cute anime cat running through a cyberpunk city at night",
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "max_sequence_length": 256,
        "num_images_per_prompt": 1
    }
    
    print("Testing Flux model...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Flux generation successful!")
            print(f"Generated image URLs: {result['response']}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def test_sd3_model():
    """Test SD3 model generation (if server is configured for SD3)"""
    url = "http://localhost:8500/api/diffusers/inference"
    
    payload = {
        "prompt": "A cute anime cat running through a cyberpunk city at night",
        "negative_prompt": "blurry, low quality",
        "num_inference_steps": 20,
        "num_images_per_prompt": 1
    }
    
    print("Testing SD3 model...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SD3 generation successful!")
            print(f"Generated image URLs: {result['response']}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def check_server_status():
    """Check if server is running"""
    try:
        response = requests.get("http://localhost:8500/api/status", timeout=10)
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Server not accessible: {e}")
        return False

if __name__ == "__main__":
    print("Testing Diffusers Server with Flux Support")
    print("=" * 50)
    
    if not check_server_status():
        print("Please start the server first with: ./run.sh")
        sys.exit(1)
    
    # Test based on command line argument
    if len(sys.argv) > 1 and sys.argv[1] == "sd3":
        success = test_sd3_model()
    else:
        success = test_flux_model()
    
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💥 Test failed!")
        sys.exit(1)