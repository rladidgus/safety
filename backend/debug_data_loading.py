import pandas as pd
import numpy as np
from pathlib import Path
import math

# Paths
BASE_DIR = Path("c:/safety-chat/backend")
DATA_DIR = BASE_DIR / "src" / "data" / "processed"

STREET_PATH = DATA_DIR / "streetlights.csv"
POLI_PATH = DATA_DIR / "police_stations.csv"
ENT_PATH = DATA_DIR / "entertainment_danger.csv"
CCTV_PATH = DATA_DIR / "cctv.csv"

# Logic from chatbot.py
def load_data(path):
    try:
        if path.exists():
            try:
                df = pd.read_csv(path, encoding='utf-8')
            except:
                df = pd.read_csv(path, encoding='cp949')
                
            col_map = {'latitude': 'lat', 'longitude': 'lon', 'y': 'lat', 'x': 'lon', '위도': 'lat', '경도': 'lon', 'address': 'addr'} 
            df = df.rename(columns=col_map)
            # Ensure coords are floats
            if 'lat' in df.columns: df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
            if 'lon' in df.columns: df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
            df = df.dropna(subset=['lat', 'lon'])
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"ERROR processing {path}: {e}")
        return pd.DataFrame()

def get_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# Load
print("Loading Data...")
STREET_DF = load_data(STREET_PATH)
POLI_DF = load_data(POLI_PATH)
CCTV_DF = load_data(CCTV_PATH)

print(f"STREET_DF: {len(STREET_DF)} rows")
print(f"POLI_DF: {len(POLI_DF)} rows")
if not STREET_DF.empty: print(f"STREET sample: \n{STREET_DF[['lat', 'lon']].head(2)}")

# Check User Coords
user_lat = 37.4851675
user_lon = 126.9296268
radius = 500

print(f"\nChecking radius {radius}m around ({user_lat}, {user_lon})...")

def check_count(df, name):
    if df.empty:
        print(f"{name}: DataFrame is Empty")
        return
    
    dists = df.apply(lambda r: get_distance(user_lat, user_lon, r['lat'], r['lon']), axis=1)
    count = len(df[dists <= radius])
    min_dist = dists.min() if len(dists) > 0 else -1
    print(f"{name}: {count} found. Closest is {min_dist:.1f}m")

check_count(CCTV_DF, "CCTV") # Benchmark (user said 311)
check_count(STREET_DF, "Streetlights") # User said 0
check_count(POLI_DF, "Police") # User said 0
