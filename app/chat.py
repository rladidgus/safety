from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import re
from .db import log_chat
from .routes_logic import find_safe_route, calculate_safety_score

# Try importing google.generativeai, handle if missing
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

router = APIRouter()

# --- Defines ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Ensure this is set in your env
if HAS_GEMINI and GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')

class ChatRequest(BaseModel):
    message: str
    start_coords: Optional[List[float]] = None # [lon, lat]
    end_coords: Optional[List[float]] = None   # [lon, lat]

class ChatResponse(BaseModel):
    response: str
    intent: str
    data: Optional[Dict[str, Any]] = None

# --- Intent Classification (Simple Rule-based) ---
def classify_intent(message: str) -> str:
    msg = message.lower()
    if any(word in msg for word in ["길", "경로", "가는법", "추천", "route", "path", "way"]):
        return "route_recommendation"
    if any(word in msg for word in ["위험", "안전", "범죄", "risk", "danger", "safe"]):
        return "safety_analysis"
    return "general_chat"

# --- Logic ---

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_msg = request.message
    intent = classify_intent(user_msg)
    bot_response = ""
    data = {}

    # 1. Route Recommendation
    if intent == "route_recommendation":
        if request.start_coords and request.end_coords:
            route_data = find_safe_route(request.start_coords, request.end_coords)
            score = route_data["properties"]["safety_score"]
            bot_response = f"추천해드린 안전 경로입니다. 예상 안전 점수는 {score}점입니다."
            data = {"route_geojson": route_data}
        else:
            bot_response = "출발지와 목적지를 설정해주시면 안전한 경로를 찾아드릴게요."
            # Client should handle this to prompt UI for location selection

    # 2. Safety Analysis (General Info)
    elif intent == "safety_analysis":
        # In a real scenario, we might analyse the current map center or a specific point
        bot_response = "현재 보고 계신 지역은 유흥업소가 밀집되어 있어 주의가 필요합니다. (예시 응답)"
        
    # 3. General Chat / Fallback (LLM)
    else:
        if HAS_GEMINI and GOOGLE_API_KEY:
            try:
                # Simple prompt wrapper
                prompt = f"You are a helpful safety assistant acting as a specialized chatbot for a Safety Map application in Seoul. Answer the following user question graciously: {user_msg}"
                response = model.generate_content(prompt)
                bot_response = response.text
            except Exception as e:
                bot_response = "죄송합니다. 현재 AI 응답을 생성할 수 없습니다."
                print(f"Gemini Error: {e}")
        else:
            bot_response = "죄송합니다. 일반 대화 기능은 현재 준비중입니다. (API Key Missing)"

    # Log to DB
    log_chat(user_msg, bot_response, intent, safety_score=data.get("route_geojson", {}).get("properties", {}).get("safety_score"))

    return ChatResponse(
        response=bot_response,
        intent=intent,
        data=data
    )
