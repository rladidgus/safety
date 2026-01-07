# app/main.py
from fastapi.responses import FileResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

# --- CORS 허용 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CSV 경로 설정 (태환님 프로젝트에 맞게 경로 수정 가능) ---
CCTV_PATH = "data/processed/cctv.csv"
STREET_PATH = "data/processed/streetlights.csv"
CONV_PATH = "data/processed/convenience_stores.csv"
ENT_PATH = "data/processed/entertainment_danger.csv"

# --- CSV 로드 ---
CCTV_DF = pd.read_csv(CCTV_PATH)
STREET_DF = pd.read_csv(STREET_PATH)
CONV_DF = pd.read_csv(CONV_PATH)
ENT_DF = pd.read_csv(ENT_PATH)

# --- 공통 포맷 변환 함수 ---
def df_to_features(df, category):
    features = []
    for _, row in df.iterrows():
        lon = row.get("lon") or row.get("longitude") or row.get("x")
        lat = row.get("lat") or row.get("latitude") or row.get("y")
        name = row.get("name") or row.get("시설명") or ""

        if pd.isna(lat) or pd.isna(lon):
            continue

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(lon), float(lat)]
            },
            "properties": {
                "category": category,
                "name": name
            }
        })
    return features

@app.get("/points")
def get_all_points():
    features = []
    features += df_to_features(CCTV_DF, "cctv")
    features += df_to_features(STREET_DF, "streetlight")
    features += df_to_features(CONV_DF, "convenience")
    features += df_to_features(ENT_DF, "entertainment")

    return {
        "type": "FeatureCollection",
        "features": features
    }
@app.get("/")
def read_root():
    # 루트로 들어오면 models/map.html 파일을 그대로 반환
    return FileResponse("models/map.html")
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory=".", html=True), name="static")