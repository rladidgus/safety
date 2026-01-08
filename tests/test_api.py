import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_chat_api():
    print("Testing /chat API...")
    url = f"{BASE_URL}/chat"
    
    # Test 1: General Greeting (Intent Check)
    payload = {"message": "안녕하세요. 안전한 길 알려주세요."}
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Test 1 Passed: {data['intent']} detected.")
        print(f"   Response: {data['response']}")
    except Exception as e:
        print(f"❌ Test 1 Failed: {e}")

    # Test 2: Dangerous Area Check
    payload = {"message": "여기 너무 위험해요."}
    try:
        response = requests.post(url, json=payload)
        data = response.json()
        print(f"✅ Test 2 Passed: {data['intent']} detected.")
        print(f"   Response: {data['response']}")
    except Exception as e:
        print(f"❌ Test 2 Failed: {e}")

    # Test 3: Route Recommendation Context
    payload = {
        "message": "안전 경로 추천해줘",
        "start_coords": [126.9780, 37.5665], # Seoul Hall
        "end_coords": [126.99, 37.57] # Random
    }
    try:
        response = requests.post(url, json=payload)
        data = response.json()
        if "data" in data and "route_geojson" in data["data"]:
            print(f"✅ Test 3 Passed: Route data returned.")
        else:
            print(f"❌ Test 3 Failed: No route data.")
    except Exception as e:
        print(f"❌ Test 3 Failed: {e}")


if __name__ == "__main__":
    try:
        test_chat_api()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure 'uvicorn app.main:app --reload' is running.")
