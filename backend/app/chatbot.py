from fastapi import APIRouter
import pandas as pd
import os
import requests
import math
import json
from pathlib import Path
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

router = APIRouter()

# --- 설정 및 API 키 ---
# 환경변수에서 API 키를 가져옵니다. .env 파일 또는 시스템 환경변수에 설정하세요.
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- 데이터 경로 설정 ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "src" / "data" / "processed"

CCTV_PATH = DATA_DIR / "cctv.csv"
STREET_PATH = DATA_DIR / "streetlights.csv"
CONV_PATH = DATA_DIR / "convenience_stores.csv"
ENT_PATH = DATA_DIR / "entertainment_danger.csv"
POLI_PATH = DATA_DIR / "police_stations.csv"
# --- 데이터 로드 및 표준화 ---
def load_data(path):
    try:
        if path.exists():
            df = pd.read_csv(path)
            col_map = {'latitude': 'lat', 'longitude': 'lon', 'y': 'lat', 'x': 'lon', '위도': 'lat', '경도': 'lon'}
            df = df.rename(columns=col_map)
            return df
        return pd.DataFrame()
    except:
        return pd.DataFrame()

CCTV_DF = load_data(CCTV_PATH)
STREET_DF = load_data(STREET_PATH)
CONV_DF = load_data(CONV_PATH)
ENT_DF = load_data(ENT_PATH)
POLI_DF = load_data(POLI_PATH)

# --- 헬퍼 함수 (거리 계산, 좌표 변환, 주소 변환) ---

def get_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # 미터
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_kakao_coords(query):
    if not query or query in ["CURRENT_LOCATION", "NONE"]: return None, None, None
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    try:
        res = requests.get("https://dapi.kakao.com/v2/local/search/keyword.json", 
                           headers=headers, params={"query": query}).json()
        if res.get('documents'):
            doc = res['documents'][0]
            return float(doc['y']), float(doc['x']), doc.get('place_name')
    except: pass
    return None, None, None

def get_address_from_kakao(lat, lng):
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    try:
        res = requests.get("https://dapi.kakao.com/v2/local/geo/coord2address.json", 
                           headers=headers, params={"x": lng, "y": lat}).json()
        documents = res.get('documents')
        if documents and len(documents) > 0:
            d = documents[0]
            road = d.get('road_address')
            if road: return road.get('address_name')
            return d.get('address', {}).get('address_name', "주소 정보 없음")
    except: pass
    return "주소 정보를 확인할 수 없는 지역입니다."

def find_nearest_cctv(lat, lon):
    if CCTV_DF.empty: return None, None
    dists = CCTV_DF.apply(lambda r: get_distance(lat, lon, r['lat'], r['lon']), axis=1)
    idx = dists.idxmin()
    return CCTV_DF.loc[idx], dists.min()

def analyze_area_stats(lat, lon, radius=500):
    stats = {"cctv": 0, "street": 0, "conv": 0, "ent": 0}
    for df, key in [(CCTV_DF, 'cctv'), (STREET_DF, 'street'), (CONV_DF, 'conv'), (ENT_DF, 'ent')]:
        if not df.empty:
            dist = df.apply(lambda r: get_distance(lat, lon, r['lat'], r['lon']), axis=1)
            stats[key] = len(df[dist <= radius])
    return stats

# --- API 요청 모델 ---
class ChatRequest(BaseModel):
    message: str
    current_lat: float
    current_lng: float

# --- 메인 챗봇 엔드포인트 ---

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # [의도 분석] 특정 지명이 포함되어 있는지 파악
    intent_prompt = f"""
    사용자 입력: '{request.message}'
    JSON으로만 답하세요:
    {{
        "intent": "NAV" | "NEAREST_CCTV" | "DANGER_ZONE" | "NEARBY_INFO",
        "target_place": "입력에 포함된 특정 지명이나 주소 (없으면 'CURRENT_LOCATION')",
        "dest": "길찾기 시 목적지 (없으면 NONE)"
    }}
    """
    intent_res = await llm.ainvoke(intent_prompt)
    nav = json.loads(intent_res.content.replace("```json", "").replace("```", "").strip())

    # [기준 좌표 설정] 주소를 입력했다면 그 지점을 기준으로, 아니면 현재 GPS 기준으로 설정
    if nav['target_place'] != "CURRENT_LOCATION":
        b_lat, b_lng, b_name = get_kakao_coords(nav['target_place'])
        if not b_lat: # 검색 실패 시 현재 위치 사용
            b_lat, b_lng, b_name = request.current_lat, request.current_lng, "현재 위치"
    else:
        b_lat, b_lng, b_name = request.current_lat, request.current_lng, "현재 위치"

    # 1. 가장 가까운 CCTV 찾기
    if nav['intent'] == "NEAREST_CCTV" or "가까운 cctv" in request.message:
        nearest, dist = find_nearest_cctv(b_lat, b_lng)
        if nearest is not None:
            cctv_addr = get_address_from_kakao(nearest['lat'], nearest['lon'])
            reply = f"### 🔍 {b_name} 주변 가장 가까운 CCTV\n\n" \
                    f"- **기준 지점:** {b_name}\n" \
                    f"- **CCTV까지 거리:** 약 {round(dist)}m\n" \
                    f"- **CCTV 주소:** {cctv_addr}\n" \
                    f"- **상세 위치:** {nearest.get('name', '정보 없음')}\n\n" \
                    f"지도상에 해당 CCTV 위치를 표시합니다."
            return {"reply": reply, "move_to": {"lat": nearest['lat'], "lng": nearest['lon']}}

    # 2. 길 안내 및 위험지역 회피 분석
    if nav['intent'] == "NAV" and nav['dest'] != "NONE":
        e_lat, e_lng, e_name = get_kakao_coords(nav['dest'])
        if e_lat:
            route_url = "https://apis-navi.kakaomobility.com/v1/directions"
            params = {"origin": f"{request.current_lng},{request.current_lat}", "destination": f"{e_lng},{e_lat}"}
            res = requests.get(route_url, headers={"Authorization": f"KakaoAK {KAKAO_API_KEY}"}, params=params).json()
            
            if "routes" in res:
                path_coords = []
                for section in res['routes'][0]['sections']:
                    for road in section['roads']:
                        v = road['vertexes']
                        for i in range(0, len(v), 2): path_coords.append((v[i+1], v[i]))
                
                # 경로상 위험 요소(유흥업소) 집계
                ent_count = 0
                for p_lat, p_lon in path_coords[::15]:
                    if not ENT_DF.empty:
                        d = ENT_DF.apply(lambda r: get_distance(p_lat, p_lon, r['lat'], r['lon']), axis=1)
                        ent_count += len(ENT_DF[d <= 50])
                
                danger_note = "⚠️ **주의:** 경로상에 유흥업소가 다수 감지되었습니다. 밝은 길로 우회하시길 권장합니다." if ent_count > 5 \
                              else "✅ **안심:** 범죄 위험 구역(유흥업소 등)을 최대한 피한 경로입니다."
                
                summary = res['routes'][0]['summary']
                reply = f"### 📍 {e_name} 안심 경로 가이드\n\n{danger_note}\n\n" \
                        f"- **예상 시간:** 약 {summary['duration']//60}분\n" \
                        f"- **이동 거리:** {round(summary['distance']/1000, 1)}km\n\n" \
                        f"안전을 위해 경로 주변 시설물을 지도에 표시했습니다."
                return {"reply": reply, "route_data": {"type": "LineString", "coordinates": [[p[1], p[0]] for p in path_coords]}}

    # 3. 위험지역/주변시설 통합 조회 (정돈된 리포트 형식)
    stats = analyze_area_stats(b_lat, b_lng)
    addr = get_address_from_kakao(b_lat, b_lng)
    
    # 위험지역 질문일 경우 상단 문구 변경
    header = "⚠️ 주변 위험 요소 분석" if (nav['intent'] == "DANGER_ZONE" or "위험" in request.message) else "🏠 주변 안전 시설 현황"
    
    reply = f"### {header}\n\n" \
            f"**[{b_name} 기준 - {addr}]**\n\n" \
            f"- **CCTV:** {stats['cctv']}개\n" \
            f"- **가로등:** {stats['street']}개\n" \
            f"- **편의점:** {stats['conv']}개\n" \
            f"- **유흥업소:** {stats['ent']}개\n\n" \
            f"주변 500m 내의 시설물 분석 결과입니다. 밤길 이동 시 참고하세요!"
    
    return {"reply": reply, "stats": stats, "move_to": {"lat": b_lat, "lng": b_lng}}
