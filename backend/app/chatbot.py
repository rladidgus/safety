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
KAKAO_API_KEY = os.getenv("KAKAO_REST_API_KEY", "")
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
CHILD_PATH = DATA_DIR / "child_protection_zones.csv"

# --- 데이터 로드 및 표준화 ---
def load_data(path):
    try:
        if path.exists():
            # 인코딩 시도 (utf-8 -> cp949)
            try:
                df = pd.read_csv(path, encoding='utf-8')
            except:
                df = pd.read_csv(path, encoding='cp949')
                
            col_map = {'latitude': 'lat', 'longitude': 'lon', 'y': 'lat', 'x': 'lon', '위도': 'lat', '경도': 'lon', 'address': 'addr', 'name': 'name', '시설명': 'name'} 
            df = df.rename(columns=col_map)
            df = df.dropna(subset=['lat', 'lon'])
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"ERROR processing {path}: {e}")
        return pd.DataFrame()

CCTV_DF = load_data(CCTV_PATH)
STREET_DF = load_data(STREET_PATH)
CONV_DF = load_data(CONV_PATH)
ENT_DF = load_data(ENT_PATH)
POLI_DF = load_data(POLI_PATH)
CHILD_DF = load_data(CHILD_PATH)

print(f"DEBUG: DATA_DIR = {DATA_DIR}")
print(f"DEBUG: Loaded STREET_DF size: {len(STREET_DF)}")
print(f"DEBUG: Loaded POLI_DF size: {len(POLI_DF)}")
print(f"DEBUG: Loaded CHILD_DF size: {len(CHILD_DF)}")

# --- 헬퍼 함수 (거리 계산, 좌표 변환, 주소 변환) ---

def get_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # 미터
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def find_nearest(lat, lon, df, k=1):
    if df.empty: return []
    try:
        df['dist'] = df.apply(lambda r: get_distance(lat, lon, r['lat'], r['lon']), axis=1)
        nearest = df.nsmallest(k, 'dist')
        results = []
        for _, row in nearest.iterrows():
            name = row.get('name', '시설')
            if pd.isna(name): name = '알 수 없음'
            results.append({"name": name, "dist": row['dist']})
        return results
    except Exception as e:
        print(f"Error finding nearest: {e}")
        return []

def get_kakao_coords(query):
    if not query or query in ["CURRENT_LOCATION", "NONE"]: return None, None, None
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    url = "https://dapi.kakao.com/v2/local/search/keyword.json"
    try:
        res = requests.get(url, headers=headers, params={"query": query}, timeout=5)
        if res.status_code == 200:
            docs = res.json().get("documents")
            if docs:
                return float(docs[0]['y']), float(docs[0]['x']), docs[0]['place_name']
    except Exception as e:
        print(f"Kakao API Error: {e}")
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



def analyze_area_stats(lat, lon, radius=500):
    stats = {"cctv": 0, "street": 0, "conv": 0, "ent": 0, "police": 0, "child": 0}
    for df, key in [(CCTV_DF, 'cctv'), (STREET_DF, 'street'), (CONV_DF, 'conv'), (ENT_DF, 'ent'), (POLI_DF, 'police'), (CHILD_DF, 'child')]:
        if not df.empty:
            if 'lat' in df.columns and 'lon' in df.columns:
                try:
                    dist = df.apply(lambda r: get_distance(lat, lon, r['lat'], r['lon']), axis=1)
                    count = len(df[dist <= radius])
                    stats[key] = count
                except Exception as e:
                    print(f"Error calculating distance for {key}: {e}")
    return stats

def analyze_wms_risk(lat, lon):
    # WMS 범죄주의구간(붉은색) 여부를 유흥업소 밀집도로 추정
    if ENT_DF.empty: return 0, "정보 없음"
    try:
        dists = ENT_DF.apply(lambda r: get_distance(lat, lon, r['lat'], r['lon']), axis=1)
        ent_count_300m = len(ENT_DF[dists <= 300])
        
        if ent_count_300m >= 15:
            return 5, "🔴 매우 위험 (범죄주의구간)"
        elif ent_count_300m >= 5:
            return 4, "🟠 위험 (주의 구간)"
        elif ent_count_300m >= 2:
            return 3, "🟡 (경계 구간)"
        else:
            return 1, "🟢 (양호 구간)"
    except:
        return 1, "정보 없음"

# --- API 요청 모델 ---
class ChatRequest(BaseModel):
    message: str
    current_lat: float
    current_lng: float

# --- 메인 챗봇 엔드포인트 ---

@router.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"DEBUG: Request received - Msg: '{request.message}', Current: ({request.current_lat}, {request.current_lng})")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # [의도 분석] 특정 지명이 포함되어 있는지 파악
    intent_prompt = f"""
    사용자 입력: '{request.message}'
    JSON으로만 답하세요:
    {{
        "intent": "DANGER_ZONE" | "NEARBY_INFO",
        "target_place": "입력에 포함된 특정 지명이나 주소 (없으면 'CURRENT_LOCATION')"
    }}
    """
    intent_res = await llm.ainvoke(intent_prompt)
    try:
        nav = json.loads(intent_res.content.replace("```json", "").replace("```", "").strip())
        print(f"DEBUG: Intent analysis result: {nav}")
    except:
        nav = {"intent": "NEARBY_INFO", "target_place": "CURRENT_LOCATION"}
        print("DEBUG: Intent analysis failed, using default")

    # [기준 좌표 설정] 주소를 입력했다면 그 지점을 기준으로, 아니면 현재 GPS 기준으로 설정
    if nav['target_place'] != "CURRENT_LOCATION":
        b_lat, b_lng, b_name = get_kakao_coords(nav['target_place'])
        print(f"DEBUG: Target place '{nav['target_place']}' -> ({b_lat}, {b_lng})")
        if not b_lat: 
            b_lat, b_lng, b_name = request.current_lat, request.current_lng, "현재 위치"
            print("DEBUG: Target place search failed, using current location")
    else:
        b_lat, b_lng, b_name = request.current_lat, request.current_lng, "현재 위치"
        print(f"DEBUG: Using current location as base: ({b_lat}, {b_lng})")


            
            
    # 2. 통합 정보 조회 (위험/안전 분석)
    stats = analyze_area_stats(b_lat, b_lng)
    risk_score, risk_label = analyze_wms_risk(b_lat, b_lng)
    
    # 가까운 중요 시설 찾기 (경찰서, 어린이 보호구역)
    nearest_police = find_nearest(b_lat, b_lng, POLI_DF)
    nearest_child = find_nearest(b_lat, b_lng, CHILD_DF)
    
    
    # 3. 메시지 구성
    
    # Police Message Construction
    police_msg = ""
    if nearest_police:
        p = nearest_police[0]
        police_msg = f"- 가까운 경찰서: {p['name']} (약 {int(p['dist'])}m)\n"
    elif stats.get('police', 0) > 0:
        police_msg = f"- 지구대/파출소: {stats['police']}개 (500m 내)\n"
    
    # Child Zone Message Construction
    child_msg = ""
    if nearest_child:
        c = nearest_child[0]
        child_msg = f"- 가까운 어린이보호구역: {c['name']} (약 {int(c['dist'])}m)\n"
    elif stats.get('child', 0) > 0:
        child_msg = f"- 어린이보호구역: {stats['child']}개 (500m 내)\n"

    # Intent Handling
    if nav['intent'] == "DANGER_ZONE" or "위험" in request.message:
        header = "🚨 주변 위험 지역 분석"
        
        if risk_score >= 4:
            advice = " 범죄 주의 구간입니다. 큰길 이용 권장."
        elif risk_score == 3:
            advice = " 주의 구간입니다. 밝은 곳으로 이동하세요."
        else:
            advice = " 비교적 안전한 지역입니다."
            
        reply = f"{header}\n\n" \
                f"{risk_label}\n\n" \
                f"{advice}\n\n" 
    
    else: # 일반 주변 정보 조회
        header = "🏠 주변 안전 시설 현황"
        
        cctv_msg = f"- CCTV: {stats['cctv']}개 (500m 내)\n" if stats.get('cctv', 0) > 0 else ""
        street_msg = f"- 가로등: {stats['street']}개\n" if stats.get('street', 0) > 0 else ""
        conv_msg = f"- 편의점: {stats['conv']}개\n" if stats.get('conv', 0) > 0 else ""

        reply = f"{header}\n\n" \
                f"{police_msg}" \
                f"{child_msg}" \
                f"{cctv_msg}" \
                f"{street_msg}" \
                f"{conv_msg}"

        # 만약 아무것도 없으면?
        if not any([police_msg, child_msg, cctv_msg, street_msg, conv_msg]):
            reply = f"{header}\n\n" \
                    "- 반경 500m 내 주요 안전 시설이 없습니다.\n"

    print(f"Stats check for ({b_lat}, {b_lng}): {stats}, Risk: {risk_score}") # Debug log
    return {"reply": reply, "stats": stats, "move_to": {"lat": b_lat, "lng": b_lng}}
