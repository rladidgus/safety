import pandas as pd
import numpy as np
from pathlib import Path
import math
import sys

# Mocking chatbot.py environment for testing
# Need to point to actual data files
BASE_DIR = Path("c:/safety-chat/backend")
DATA_DIR = BASE_DIR / "src" / "data" / "processed"

CCTV_PATH = DATA_DIR / "cctv.csv"
STREET_PATH = DATA_DIR / "streetlights.csv"
CONV_PATH = DATA_DIR / "convenience_stores.csv"
ENT_PATH = DATA_DIR / "entertainment_danger.csv"
POLI_PATH = DATA_DIR / "police_stations.csv"

def load_data(path):
    try:
        if path.exists():
            try:
                df = pd.read_csv(path, encoding='utf-8')
            except:
                df = pd.read_csv(path, encoding='cp949')
            col_map = {'latitude': 'lat', 'longitude': 'lon', 'y': 'lat', 'x': 'lon', '위도': 'lat', '경도': 'lon', 'address': 'addr'} 
            df = df.rename(columns=col_map)
            return df
        return pd.DataFrame()
    except Exception as e:
        print(f"ERROR: {e}")
        return pd.DataFrame()

print("Loading data...")
CCTV_DF = load_data(CCTV_PATH)
STREET_DF = load_data(STREET_PATH)
ENT_DF = load_data(ENT_PATH)

def get_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def analyze_area_stats(lat, lon, radius=500):
    stats = {"cctv": 0, "ent": 0}
    
    # CCTV
    if not CCTV_DF.empty:
        d = CCTV_DF.apply(lambda r: get_distance(lat, lon, r['lat'], r['lon']), axis=1)
        stats['cctv'] = len(CCTV_DF[d <= radius])
        
    # ENT
    if not ENT_DF.empty:
        d = ENT_DF.apply(lambda r: get_distance(lat, lon, r['lat'], r['lon']), axis=1)
        stats['ent'] = len(ENT_DF[d <= radius])
    
    return stats

# Test Cases
# 1. Gangnam Station (Crowded)
gangnam = (37.498095, 127.027610)
# 2. Bukhansan (Quiet)
mountain = (37.660725, 126.994273)
# 3. Sillim
sillim = (37.484258, 126.929764)

print(f"\nTesting Gangnam {gangnam}:")
print(analyze_area_stats(*gangnam))

print(f"\nTesting Mountain {mountain}:")
print(analyze_area_stats(*mountain))

print(f"\nTesting Sillim {sillim}:")
print(analyze_area_stats(*sillim))
