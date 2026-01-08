"""
OSM 기반 도로망 그래프 구축 모듈
- OpenStreetMap에서 서울시 도보 네트워크 다운로드
- 학습된 시설물 데이터로 안전 가중치 적용
- 시간대별 가로등 점등 상태 반영
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import osmnx as ox
    import networkx as nx
    from scipy.spatial import cKDTree
    LIBS_AVAILABLE = True
except ImportError as e:
    LIBS_AVAILABLE = False
    print(f"⚠️ 필요한 패키지: pip install osmnx networkx scipy")

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"


def download_seoul_network(place: str = "Seoul, South Korea") -> nx.MultiDiGraph:
    """
    OSM에서 서울시 도보 네트워크 다운로드
    
    Args:
        place: 지역명 (기본: 서울)
    
    Returns:
        NetworkX 그래프
    """
    if not LIBS_AVAILABLE:
        raise ImportError("osmnx 패키지 필요: pip install osmnx")
    
    print(f"🗺️ OSM에서 도로망 다운로드 중: {place}")
    print("   (처음 실행 시 5-10분 소요될 수 있습니다...)")
    
    # OSM 설정
    ox.settings.log_console = False
    ox.settings.use_cache = True
    
    # 도보 네트워크 다운로드
    G = ox.graph_from_place(place, network_type="walk")
    
    print(f"✅ 다운로드 완료!")
    print(f"   - 노드: {G.number_of_nodes():,}")
    print(f"   - 엣지: {G.number_of_edges():,}")
    
    return G


def load_facility_data() -> Dict[str, pd.DataFrame]:
    """전처리된 시설물 데이터 로드"""
    facilities = {}
    
    files = {
        'streetlight': 'streetlights.csv',
        'cctv': 'cctv.csv',
        'police': 'police_stations.csv',
        'convenience': 'convenience_stores.csv',
        'entertainment': 'entertainment_danger.csv',
        'child_zone': 'child_protection_zones.csv'
    }
    
    for key, filename in files.items():
        filepath = PROCESSED_DIR / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            facilities[key] = df
            print(f"   ✅ {key}: {len(df):,} 건")
        else:
            facilities[key] = pd.DataFrame()
    
    return facilities


def load_streetlight_schedule() -> Dict[str, int]:
    """가로등 점소등 시간 로드"""
    filepath = DATA_DIR / "서울시 가로등 점소등 시간 현황.csv"
    
    if not filepath.exists():
        return {'on_hour': 18, 'off_hour': 6}
    
    try:
        df = pd.read_csv(filepath, encoding='cp949')
        on_cols = [c for c in df.columns if '점등' in c]
        off_cols = [c for c in df.columns if '소등' in c]
        
        return {'on_hour': 18, 'off_hour': 6}
    except:
        return {'on_hour': 18, 'off_hour': 6}


def is_streetlight_on(hour: int, schedule: Dict[str, int]) -> bool:
    """가로등 점등 여부 확인"""
    on_hour = schedule.get('on_hour', 18)
    off_hour = schedule.get('off_hour', 6)
    
    if on_hour > off_hour:
        return hour >= on_hour or hour < off_hour
    else:
        return on_hour <= hour < off_hour


def build_facility_trees(facilities: Dict[str, pd.DataFrame]) -> Dict[str, cKDTree]:
    """시설물 KDTree 생성"""
    trees = {}
    
    for key, df in facilities.items():
        if len(df) > 0 and 'latitude' in df.columns:
            coords = df[['latitude', 'longitude']].values
            trees[key] = cKDTree(coords)
    
    return trees


def count_nearby(point: np.ndarray, tree: Optional[cKDTree], radius_m: float) -> int:
    """지점 근처 시설물 개수"""
    if tree is None:
        return 0
    radius_deg = radius_m / 111000
    return len(tree.query_ball_point(point, radius_deg))


def get_highway_danger_adjustment(highway_type: str) -> float:
    """
    도로 유형별 위험도 조정값
    큰 도로일수록 안전 (음수), 골목/소로는 위험 (양수)
    
    OSM highway 타입 참고:
    - trunk, primary, secondary: 대로 (가장 안전)
    - tertiary: 중간 도로
    - residential, unclassified: 주거지 도로
    - service, alley, path, footway: 골목/소로 (상대적 위험)
    """
    # highway가 리스트인 경우 첫 번째 값 사용
    if isinstance(highway_type, list):
        highway_type = highway_type[0] if highway_type else 'unknown'
    
    highway_type = str(highway_type).lower()
    
    # 대로 - 사람이 많고 조명이 밝아 안전
    if highway_type in ['trunk', 'trunk_link', 'primary', 'primary_link']:
        return -0.25  # 매우 안전
    elif highway_type in ['secondary', 'secondary_link']:
        return -0.20  # 안전
    elif highway_type in ['tertiary', 'tertiary_link']:
        return -0.15  # 비교적 안전
    
    # 일반 도로
    elif highway_type in ['residential', 'unclassified', 'living_street']:
        return 0.0  # 보통
    
    # 골목/소로 - 상대적으로 위험
    elif highway_type in ['service', 'alley']:
        return 0.15  # 위험
    elif highway_type in ['path', 'footway', 'pedestrian', 'steps']:
        return 0.10  # 약간 위험 (보행자 전용이라 차는 없지만 어두울 수 있음)
    elif highway_type in ['cycleway']:
        return 0.05  # 약간 위험
    
    # 알 수 없는 유형
    else:
        return 0.05  # 약간 위험 (보수적)


def calculate_danger_score(
    lat: float, lon: float,
    trees: Dict[str, cKDTree],
    hour: int,
    streetlight_on: bool,
    highway_type: str = 'unknown'
) -> float:
    """위험도 계산 (0~1, 높을수록 위험)"""
    point = np.array([lat, lon])
    
    # 안전 요소
    streetlight_count = count_nearby(point, trees.get('streetlight'), 50)
    cctv_count = count_nearby(point, trees.get('cctv'), 50)
    police_nearby = count_nearby(point, trees.get('police'), 500) > 0
    convenience_count = count_nearby(point, trees.get('convenience'), 100)
    child_zone = count_nearby(point, trees.get('child_zone'), 200) > 0
    
    # 위험 요소
    entertainment_count = count_nearby(point, trees.get('entertainment'), 100)
    
    # 위험도 계산
    danger = 0.5  # 기본값
    
    # ★ 도로 유형 반영 (가장 중요한 요소)
    danger += get_highway_danger_adjustment(highway_type)
    
    # 안전 요소 (위험도 감소)
    if streetlight_on:
        danger -= min(streetlight_count * 0.03, 0.15)
    danger -= min(cctv_count * 0.02, 0.1)
    if police_nearby:
        danger -= 0.1
    danger -= min(convenience_count * 0.02, 0.1)
    if child_zone:
        danger -= 0.05
    
    # 위험 요소 (위험도 증가)
    danger += min(entertainment_count * 0.04, 0.2)
    
    # 야간 추가 위험 (골목일수록 야간 위험도 더 증가)
    if hour < 6 or hour >= 22:
        danger += 0.15
    elif hour >= 21:
        danger += 0.1
    
    return np.clip(danger, 0.1, 0.9)


def apply_safety_weights(
    G: nx.MultiDiGraph,
    facilities: Dict[str, pd.DataFrame],
    hour: int = None
) -> nx.MultiDiGraph:
    """
    OSM 그래프에 안전 가중치 적용
    """
    current_hour = hour if hour is not None else datetime.now().hour
    
    # 가로등 점등 상태
    sl_schedule = load_streetlight_schedule()
    streetlight_on = is_streetlight_on(current_hour, sl_schedule)
    
    print(f"\n⏰ 시간: {current_hour}시 (가로등: {'ON' if streetlight_on else 'OFF'})")
    
    # 시설물 KDTree 생성
    print("\n🔍 시설물 인덱싱...")
    trees = build_facility_trees(facilities)
    
    # 엣지에 가중치 적용
    print("\n🔄 안전 가중치 계산 중...")
    edges = list(G.edges(keys=True, data=True))
    
    for u, v, key, data in tqdm(edges, desc="엣지 가중치"):
        # 엣지 중심 좌표
        u_lat, u_lon = G.nodes[u]['y'], G.nodes[u]['x']
        v_lat, v_lon = G.nodes[v]['y'], G.nodes[v]['x']
        center_lat = (u_lat + v_lat) / 2
        center_lon = (u_lon + v_lon) / 2
        
        # 도로 길이
        length = data.get('length', 100)
        
        # 도로 유형 (OSM highway 속성)
        highway_type = data.get('highway', 'unknown')
        
        # 위험도 계산 (도로 유형 포함)
        danger_score = calculate_danger_score(
            center_lat, center_lon,
            trees, current_hour, streetlight_on,
            highway_type=highway_type
        )
        
        # 안전 가중치 (위험할수록 높음)
        safety_weight = length * (1 + danger_score * 2)
        
        # 그래프에 저장
        G.edges[u, v, key]['safety_weight'] = safety_weight
        G.edges[u, v, key]['danger_score'] = danger_score
        G.edges[u, v, key]['safety_score'] = int(100 * (1 - danger_score))
    
    # 메타데이터
    G.graph['hour'] = current_hour
    G.graph['streetlight_on'] = streetlight_on
    G.graph['created_at'] = datetime.now().isoformat()
    G.graph['source'] = 'OpenStreetMap'
    
    return G


def save_graph(G: nx.MultiDiGraph, filename: str = "seoul_osm_safety_graph"):
    """그래프 저장"""
    MODEL_DIR.mkdir(exist_ok=True)
    
    filepath = MODEL_DIR / f"{filename}.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(G, f)
    print(f"   ✅ 그래프 저장: {filepath}")
    
    return filepath


def load_graph(filename: str = "seoul_osm_safety_graph") -> nx.MultiDiGraph:
    """그래프 로드"""
    filepath = MODEL_DIR / f"{filename}.pkl"
    
    if not filepath.exists():
        raise FileNotFoundError(f"그래프 파일 없음: {filepath}")
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def print_graph_stats(G: nx.MultiDiGraph):
    """그래프 통계 출력"""
    edges = list(G.edges(data=True))
    
    dangers = [e[2].get('danger_score', 0.5) for e in edges]
    lengths = [e[2].get('length', 0) for e in edges]
    
    print(f"\n📊 그래프 통계:")
    print(f"   - 노드: {G.number_of_nodes():,}")
    print(f"   - 엣지: {G.number_of_edges():,}")
    print(f"   - 시간: {G.graph.get('hour', 'N/A')}시")
    print(f"   - 가로등: {'ON' if G.graph.get('streetlight_on', False) else 'OFF'}")
    print(f"   - 데이터 소스: {G.graph.get('source', 'Unknown')}")
    
    print(f"\n📈 위험도 분포:")
    print(f"   - 평균: {np.mean(dangers):.3f}")
    print(f"   - 최소: {np.min(dangers):.3f}")
    print(f"   - 최대: {np.max(dangers):.3f}")
    
    print(f"\n📏 도로 길이:")
    print(f"   - 총: {sum(lengths)/1000:.1f} km")


def main(place: str = "Seoul, South Korea", hour: int = None):
    """메인 실행"""
    print("=" * 60)
    print("🚀 OSM 기반 안전 도로망 구축")
    print("=" * 60)
    
    # 1. OSM 네트워크 다운로드
    G = download_seoul_network(place)
    
    # 2. 시설물 데이터 로드
    print("\n📂 시설물 데이터 로드...")
    facilities = load_facility_data()
    
    # 3. 안전 가중치 적용
    G = apply_safety_weights(G, facilities, hour=hour)
    
    # 4. 통계 출력
    print_graph_stats(G)
    
    # 5. 저장
    print("\n💾 그래프 저장...")
    save_graph(G)
    
    print("\n" + "=" * 60)
    print("✅ OSM 기반 도로망 구축 완료!")
    print("=" * 60)
    
    return G


if __name__ == "__main__":
    # 서울 전체, 야간 (22시)
    main("Seoul, South Korea", hour=22)
