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
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

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
    
    Args:
        G: 원본 도로망 그래프
        facilities: 시설물 데이터 딕셔너리
        ml_model: 학습된 SafetyMLModel
        hour: 시간 (0-23). None이면 현재 시간 사용
        verbose: 진행 상황 출력 여부
    
    Returns:
        ML 기반 안전 가중치가 적용된 그래프
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
    
    print(f"\n⏰ 시간대: {period} ({current_hour}시)")
    print(f"   시간대 위험도: {hour_danger:.3f} (학습된 값)")
    print(f"   요일 위험도: {day_danger:.3f} (학습된 값)")
    
    print("\n🤖 향상된 ML 기반 가중치 모드")
    print(f"   사용 피처 수: {len(ml_model.feature_columns)}")
    
    # 상위 5개 피처 중요도만 출력
    if ml_model.feature_importance:
        print(f"   주요 피처 중요도:")
        sorted_imp = sorted(ml_model.feature_importance.items(), 
                           key=lambda x: x[1], reverse=True)[:5]
        for name, imp in sorted_imp:
            print(f"      {name}: {imp:.4f}")
    
    print("\n🔄 향상된 피처 추출기 초기화 중...")
    
    # 향상된 피처 추출기 생성
    extractor = EnhancedFeatureExtractor(facilities)
    
    print("\n🔄 ML 예측으로 안전 가중치 계산 중...")
    
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
        G.edges[u, v, key]['streetlight_count'] = features['streetlight_count']
        G.edges[u, v, key]['cctv_count'] = features['cctv_count']
        G.edges[u, v, key]['police_nearby'] = features['police_nearby']
        G.edges[u, v, key]['convenience_count'] = features['convenience_count']
        G.edges[u, v, key]['entertainment_count'] = features['entertainment_count']
        G.edges[u, v, key]['isolation_score'] = features['isolation_score']
        G.edges[u, v, key]['hour_danger'] = hour_danger
        G.edges[u, v, key]['day_danger'] = day_danger
        G.edges[u, v, key]['ml_applied'] = True
    
    # 그래프에 메타데이터 저장
    G.graph['time_period'] = period
    G.graph['hour'] = current_hour
    G.graph['day_of_week'] = current_day
    G.graph['hour_danger'] = hour_danger
    G.graph['day_danger'] = day_danger
    G.graph['weight_mode'] = 'ML_Enhanced_TimeAware'
    
    return G


def save_graph(G: nx.MultiDiGraph, filename: str = "safety_graph"):
    """그래프 저장"""
    MODEL_DIR.mkdir(exist_ok=True)
    
    filepath_graphml = MODEL_DIR / f"{filename}.graphml"
    ox.save_graphml(G, filepath_graphml)
    print(f"   ✅ GraphML 저장: {filepath_graphml}")
    
    filepath_pkl = MODEL_DIR / f"{filename}.pkl"
    with open(filepath_pkl, 'wb') as f:
        pickle.dump(G, f)
    print(f"   ✅ Pickle 저장: {filepath_pkl}")


def load_graph(filename: str = "safety_graph") -> nx.MultiDiGraph:
    """저장된 그래프 로드"""
    filepath_pkl = MODEL_DIR / f"{filename}.pkl"
    
    if filepath_pkl.exists():
        with open(filepath_pkl, 'rb') as f:
            return pickle.load(f)
    
    filepath_graphml = MODEL_DIR / f"{filename}.graphml"
    if filepath_graphml.exists():
        return ox.load_graphml(filepath_graphml)
    
    raise FileNotFoundError(f"그래프 파일을 찾을 수 없습니다: {filename}")


def print_graph_stats(G: nx.MultiDiGraph):
    """그래프 통계 출력"""
    edges = list(G.edges(data=True))
    
    safety_scores = [e[2].get('safety_score', 50) for e in edges]
    dangers = [e[2].get('predicted_danger', 0.5) for e in edges]
    streetlights = [e[2].get('streetlight_count', 0) for e in edges]
    cctvs = [e[2].get('cctv_count', 0) for e in edges]
    conveniences = [e[2].get('convenience_count', 0) for e in edges]
    entertainments = [e[2].get('entertainment_count', 0) for e in edges]
    
    print("\n📊 그래프 통계:")
    print(f"   - 총 노드: {G.number_of_nodes():,}")
    print(f"   - 총 엣지: {G.number_of_edges():,}")
    print(f"   - 가중치 모드: {G.graph.get('weight_mode', 'Rule-based')}")
    print(f"   - 시간대: {G.graph.get('time_period', 'N/A')} ({G.graph.get('hour', 'N/A')}시)")
    
    print(f"\n📈 ML 예측 위험도 분포:")
    print(f"   - 평균: {np.mean(dangers):.3f}")
    print(f"   - 최소: {np.min(dangers):.3f} (가장 안전)")
    print(f"   - 최대: {np.max(dangers):.3f} (가장 위험)")
    
    print(f"\n📈 안전 점수 분포:")
    print(f"   - 평균: {np.mean(safety_scores):.1f}")
    print(f"   - 최소: {np.min(safety_scores):.1f}")
    print(f"   - 최대: {np.max(safety_scores):.1f}")
    
    print(f"\n🔦 시설물 매칭 현황:")
    print(f"   - 가로등 매칭 edge: {sum(1 for s in streetlights if s > 0):,}")
    print(f"   - CCTV 매칭 edge: {sum(1 for c in cctvs if c > 0):,}")
    print(f"   - 편의점 매칭 edge: {sum(1 for c in conveniences if c > 0):,}")
    print(f"   - 유흥업소 매칭 edge: {sum(1 for e in entertainments if e > 0):,}")


def main(place: str = "Gangnam-gu, Seoul, South Korea", hour: int = None):
    """
    메인 실행 함수 (ML 기반)
    
    Args:
        place: 도로망을 가져올 지역 (기본: 강남구)
        hour: 시간대 (0-23). None이면 현재 시간 사용
    """
    print("=" * 60)
    print("🚀 안심 길 안내 - 도로망 그래프 구축 (ML 기반)")
    print("=" * 60)
    
    # 1. 도로망 가져오기
    G = get_road_network(place, network_type="walk")
    
    # 2. 시설물 데이터 로드
    print("\n📂 시설물 데이터 로드 중...")
    facilities = load_facility_data()
    
    # 3. ML 모델 로드
    print("\n📂 ML 모델 로드 중...")
    ml_model = load_ml_model()
    
    if ml_model is None:
        print("\n❌ ML 모델이 없습니다.")
        print("💡 먼저 다음 명령을 실행하세요:")
        print("   python src/ml_trainer.py")
        return None
    
    # 4. ML 가중치 그래프 구축
    G = build_weighted_graph_ml(G, facilities, ml_model, hour=hour)
    
    # 5. 통계 출력
    print_graph_stats(G)
    
    # 6. 그래프 저장
    print("\n💾 그래프 저장 중...")
    save_graph(G, "safety_graph")
    
    print("\n" + "=" * 60)
    print("✅ ML 기반 도로망 그래프 구축 완료!")
    print("=" * 60)
    
    return G


if __name__ == "__main__":
    # 테스트: 강남구, 야간 (23시)
    main("Gangnam-gu, Seoul, South Korea", hour=23)
