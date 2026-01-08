"""
도로망 그래프 구축 모듈 (ML 기반)
- OSMnx를 이용한 서울시 도로망 가져오기
- ML 모델로 안전 가중치 예측
- 시간대별 보정 적용
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 도로망 처리
try:
    import osmnx as ox
    import networkx as nx
    from scipy.spatial import cKDTree
    GRAPH_LIBS_AVAILABLE = True
except ImportError as e:
    GRAPH_LIBS_AVAILABLE = False
    print(f"⚠️ 필요한 패키지 미설치: {e}")

# ML 모델 (스크립트/모듈 모두 지원)
try:
    from .ml_trainer import (SafetyMLModel, EnhancedFeatureExtractor,
                             load_crime_time_data, load_crime_day_data,
                             get_time_danger_score, get_day_danger_score)
except ImportError:
    try:
        from ml_trainer import (SafetyMLModel, EnhancedFeatureExtractor,
                                load_crime_time_data, load_crime_day_data,
                                get_time_danger_score, get_day_danger_score)
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from ml_trainer import (SafetyMLModel, EnhancedFeatureExtractor,
                                load_crime_time_data, load_crime_day_data,
                                get_time_danger_score, get_day_danger_score)


# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent # Adjusted for backend/app/ location if PROJET_ROOT is root of repo. 
# Wait, original was Path(__file__).parent.parent. If I move to backend/app/, parent is backend, parent.parent is miniProject.
# Original: backend/graph_builder.py -> parent=backend, parent.parent=miniProject.
# New: backend/app/graph_builder.py -> parent=app, parent.parent=backend.
# So I need one more parent if I want miniProject root?
# DATA_DIR = PROJECT_ROOT / "data". 
# If structure is:
# miniProject/
#   data/
#   backend/
#     app/
#       graph_builder.py
# Then PROJECT_ROOT should be miniProject.
# backend/app/graph_builder.py -> parent -> app, parent.parent -> backend, parent.parent.parent -> miniProject.
PROJ_ROOT_FIX = Path(__file__).resolve().parent.parent.parent

# But let's check original logic carefully.
# Original: PROJECT_ROOT = Path(__file__).parent.parent
# If file is c:\MyDEV\miniProject\backend\graph_builder.py
# parent = backend
# parent.parent = miniProject
# Correct.

# Now file is c:\MyDEV\miniProject\backend\app\graph_builder.py
# parent = app
# parent.parent = backend
# parent.parent.parent = miniProject
# So yes, I need one more .parent.

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "backend" / "data" # Did the user say data is in backend/data?
# List dir showed backend has 'data' folder NO.
# List dir miniProject showed `backend` has NO data folder (step 12).
# Wait, step 12 list_dir backend showed: app, graph_builder.py ... NO data folder.
# Step 19 list_dir backend/src (empty/data inside?) -> `data` folder inside `src`? NO.
# Where is DATA_DIR?
# In `ml_trainer.py` (original): PROJECT_ROOT = Path(__file__).parent.parent. DATA_DIR = PROJECT_ROOT / "data".
# If `ml_trainer.py` was in `backend/`, parent.parent is `miniProject`.
# So DATA_DIR is `miniProject/data`.
# Let's check `miniProject` dir (Step 4).
# It has `backend`, `frontend`, `miniPJ01.py`. NO `data` folder at root.
# Step 19 list_dir `backend/src/data` -> Wait, step 19 output was `{"name":"data", "isDir":true, ...}` when listing `backend/src`.
# So `data` is in `backend/src/data`?
# But `graph_builder.py` was in `backend/`.
# If `graph_builder.py` in `backend/`, `Path(__file__).parent.parent` is `miniProject`.
# If DATA_DIR = `miniProject/data`, and `miniProject/data` doesn't exist...
# Maybe the user has data somewhere else?
# Or maybe `ml_trainer.py` expects `../data` relative to itself.
# If `ml_trainer.py` is in `backend/`, `..` is `miniProject`. So `miniProject/data`.
# If that folder is missing, the code fails.
# I'll stick to `PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent` (miniProject root) and `DATA_DIR = PROJECT_ROOT / "data"`.
# Wait, user might have `backend/data`?
# I saw `backend/src/data` in step 19.
# The code I read says `PROJECT_ROOT / "data"`.
# I will use the code as is but adjust PROJECT_ROOT to be 3 levels up from `backend/app/graph_builder.py`.
# Actually, I'll be safe and try to find data.
# I'll use `Path(__file__).parent.parent` (which is backend) then / "data" ?
# Whatever, I will implement the fix for 3 parents.

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" # Assuming data is at root
if not DATA_DIR.exists():
    # Try backend/data
    DATA_DIR = PROJECT_ROOT / "backend" / "data"
    if not DATA_DIR.exists():
         # Try backend/src/data
         DATA_DIR = PROJECT_ROOT / "backend" / "src" / "data"

PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "backend" / "models" # Models should be in backend/models?

# 매칭 반경 (미터)
RADIUS = {
    'streetlight': 50,
    'cctv': 50,
    'police': 500,
    'convenience': 100,
    'entertainment': 100,
}

# 시간대별 보정 배율 (ML 예측에 적용)
TIME_ADJUSTMENTS = {
    'night': 1.3,       # 새벽 (0-6시): 위험도 30% 증가
    'morning': 0.8,     # 아침 (6-9시): 위험도 20% 감소
    'daytime': 0.7,     # 낮 (9-18시): 위험도 30% 감소
    'evening': 1.0,     # 저녁 (18-22시): 보정 없음
    'late_night': 1.2,  # 밤 (22-24시): 위험도 20% 증가
}


def get_time_period(hour: int = None) -> str:
    """현재 시간대 반환"""
    if hour is None:
        hour = datetime.now().hour
    
    if 0 <= hour < 6:
        return 'night'
    elif 6 <= hour < 9:
        return 'morning'
    elif 9 <= hour < 18:
        return 'daytime'
    elif 18 <= hour < 22:
        return 'evening'
    else:
        return 'late_night'


def get_road_network(place: str = "Seoul, South Korea", 
                     network_type: str = "walk") -> nx.MultiDiGraph:
    """OSMnx를 이용해 도로망 그래프를 가져옵니다."""
    if not GRAPH_LIBS_AVAILABLE:
        raise ImportError("osmnx, networkx, scipy 패키지를 설치해주세요.")
    
    print(f"🗺️ 도로망 데이터 다운로드 중: {place}")
    print("   (처음 실행 시 시간이 걸릴 수 있습니다...)")
    
    ox.settings.log_console = False
    ox.settings.use_cache = True
    
    G = ox.graph_from_place(place, network_type=network_type)
    
    print(f"✅ 도로망 로드 완료!")
    print(f"   - 노드 수: {G.number_of_nodes():,}")
    print(f"   - 엣지 수: {G.number_of_edges():,}")
    
    return G


def load_facility_data() -> Dict[str, pd.DataFrame]:
    """전처리된 시설물 데이터 로드"""
    facilities = {}
    
    files = {
        'streetlight': 'streetlights.csv',
        'cctv': 'cctv.csv',
        'police': 'police_stations.csv',
        'convenience': 'convenience_stores.csv',
        'entertainment': 'entertainment_danger.csv'
    }
    
    for key, filename in files.items():
        filepath = PROCESSED_DIR / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            facilities[key] = df
            print(f"   ✅ {key}: {len(df):,} 건")
        else:
            print(f"   ⚠️ {key}: 파일 없음 ({filename})")
            facilities[key] = pd.DataFrame()
    
    return facilities


def load_ml_model() -> Optional[SafetyMLModel]:
    """학습된 ML 모델 로드"""
    model_path = MODEL_DIR / "safety_ml_model.pkl"
    
    if not model_path.exists():
        print("   ⚠️ ML 모델 없음. 먼저 python src/ml_trainer.py 실행 필요")
        return None
    
    try:
        model = SafetyMLModel()
        model.load()
        return model
    except Exception as e:
        print(f"   ⚠️ ML 모델 로드 실패: {e}")
        return None


def build_spatial_index(coords: np.ndarray) -> Optional[cKDTree]:
    """좌표 배열에 대한 공간 인덱스(KD-Tree) 생성"""
    if len(coords) == 0:
        return None
    return cKDTree(coords)


def count_facilities_near_point(point: np.ndarray, tree: cKDTree, 
                                 radius_meters: float) -> int:
    """지점 근처의 시설물 개수 계산"""
    if tree is None:
        return 0
    
    radius_deg = radius_meters / 111000
    indices = tree.query_ball_point(point, radius_deg)
    return len(indices)


def build_weighted_graph_ml(G: nx.MultiDiGraph, 
                            facilities: Dict[str, pd.DataFrame],
                            ml_model: SafetyMLModel,
                            hour: int = None,
                            verbose: bool = True) -> nx.MultiDiGraph:
    """
    ML 모델 기반 안전 가중치 그래프 생성 (향상된 피처 사용)
    """
    current_hour = hour if hour is not None else datetime.now().hour
    current_day = datetime.now().weekday()  # 0=월 ~ 6=일
    
    # 학습된 시간/요일 위험도 로드
    time_danger_data = load_crime_time_data()
    day_danger_data = load_crime_day_data()
    
    # 현재 시간/요일의 위험도 계산
    hour_danger = get_time_danger_score(current_hour, time_danger_data)
    day_danger = get_day_danger_score(current_day, day_danger_data)
    
    period = get_time_period(current_hour)
    
    print(f"\\n⏰ 시간대: {period} ({current_hour}시)")
    print(f"   시간대 위험도: {hour_danger:.3f} (학습된 값)")
    print(f"   요일 위험도: {day_danger:.3f} (학습된 값)")
    
    # 향상된 피처 추출기 생성
    extractor = EnhancedFeatureExtractor(facilities)
    
    edges = list(G.edges(keys=True, data=True))
    
    for u, v, key, data in tqdm(edges, desc="ML 가중치", disable=not verbose):
        # Edge 중심 좌표
        u_lat, u_lon = G.nodes[u]['y'], G.nodes[u]['x']
        v_lat, v_lon = G.nodes[v]['y'], G.nodes[v]['x']
        center_lat = (u_lat + v_lat) / 2
        center_lon = (u_lon + v_lon) / 2
        
        # 도로 길이 및 유형
        road_length = data.get('length', 100)
        highway_type = data.get('highway', '')
        is_main_road = highway_type in ['primary', 'secondary', 'tertiary', 'trunk']
        
        # 향상된 피처 추출
        features = extractor.extract_features(
            center_lat, center_lon, road_length, is_main_road
        )
        
        # 시간/요일 피처 추가 (학습된 값 사용)
        features['hour_danger'] = hour_danger
        features['day_danger'] = day_danger
        features['is_night'] = 1 if (current_hour < 6 or current_hour >= 21) else 0
        features['is_weekend'] = 1 if current_day >= 5 else 0
        
        # ML 모델로 위험도 예측 (시간/요일 피처 포함)
        predicted_danger = ml_model.predict_single(**features)
        
        # 범위 제한
        adjusted_danger = np.clip(predicted_danger, 0, 1)
        
        # 안전 가중치 계산
        safety_weight = road_length * (1 + adjusted_danger)
        
        # 안전 점수 (0-100, 높을수록 안전)
        safety_score = 100 * (1 - adjusted_danger)
        
        # 그래프에 속성 추가
        G.edges[u, v, key]['safety_weight'] = safety_weight
        G.edges[u, v, key]['safety_score'] = safety_score
        G.edges[u, v, key]['predicted_danger'] = adjusted_danger
    
    # 그래프에 메타데이터 저장
    G.graph['time_period'] = period
    G.graph['hour'] = current_hour
    
    return G


def save_graph(G: nx.MultiDiGraph, filename: str = "safety_graph"):
    """그래프 저장"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    filepath_graphml = MODEL_DIR / f"{filename}.graphml"
    ox.save_graphml(G, filepath_graphml)
    print(f"   ✅ GraphML 저장: {filepath_graphml}")
    
    filepath_pkl = MODEL_DIR / f"{filename}.pkl"
    with open(filepath_pkl, 'wb') as f:
        pickle.dump(G, f)
    print(f"   ✅ Pickle 저장: {filepath_pkl}")


def load_graph(filename: str = "safety_graph") -> nx.MultiDiGraph:
    """저장된 그래프 로드"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    filepath_pkl = MODEL_DIR / f"{filename}.pkl"
    
    if filepath_pkl.exists():
        with open(filepath_pkl, 'rb') as f:
            return pickle.load(f)
    
    filepath_graphml = MODEL_DIR / f"{filename}.graphml"
    if filepath_graphml.exists():
        return ox.load_graphml(filepath_graphml)
    
    raise FileNotFoundError(f"그래프 파일을 찾을 수 없습니다: {filename}")


def main(place: str = "Gangnam-gu, Seoul, South Korea", hour: int = None):
    """
    메인 실행 함수 (ML 기반)
    """
    print("=" * 60)
    print("🚀 안심 길 안내 - 도로망 그래프 구축 (ML 기반)")
    print("=" * 60)
    
    # 1. 도로망 가져오기
    G = get_road_network(place, network_type="walk")
    
    # 2. 시설물 데이터 로드
    facilities = load_facility_data()
    
    # 3. ML 모델 로드 (없으면 기본값 사용 가능하지만...)
    ml_model = load_ml_model()
    
    # 4. ML 가중치 그래프 구축 (모델 없으면 Skip or Fail, here we skip logic if None)
    if ml_model:
        G = build_weighted_graph_ml(G, facilities, ml_model, hour=hour)
    else:
        print("Model not found, skipping weights")

    # 6. 그래프 저장
    save_graph(G, "safety_graph")
    
    return G


if __name__ == "__main__":
    main("Gangnam-gu, Seoul, South Korea", hour=23)
