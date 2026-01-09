"""
OSM 기반 안전 경로 탐색 모듈
- 실시간 시간대 기반 안전 점수 계산
- 최단 경로 vs 최안전 경로 비교
- 서울 전체 어디서나 경로 탐색 가능
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    import osmnx as ox
    import networkx as nx
    LIBS_AVAILABLE = True
except ImportError as e:
    LIBS_AVAILABLE = False
    print(f"⚠️ 필요한 패키지: pip install osmnx networkx")

# 프로젝트 경로 (backend/app/route_finder.py 기준)
# parent -> app, parent.parent -> backend
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


def load_graph(filename: str = "seoul_osm_safety_graph") -> nx.MultiDiGraph:
    """저장된 OSM 그래프 로드"""
    filepath = MODEL_DIR / f"{filename}.pkl"
    
    if not filepath.exists():
        raise FileNotFoundError(f"그래프 파일 없음: {filepath}")
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def is_streetlight_on(hour: int) -> bool:
    """가로등 점등 여부 (18시~6시)"""
    return hour >= 18 or hour < 6


def get_time_danger_adjustment(hour: int) -> float:
    """
    시간대별 위험도 조정값
    
    Returns:
        조정값 (양수: 위험 증가, 음수: 위험 감소)
    """
    if 0 <= hour < 6:      # 새벽 (가장 위험)
        return 0.20
    elif 6 <= hour < 9:    # 출근 시간
        return -0.10
    elif 9 <= hour < 18:   # 낮 (가장 안전)
        return -0.15
    elif 18 <= hour < 21:  # 저녁
        return 0.0
    elif 21 <= hour < 22:  # 밤
        return 0.10
    else:                  # 심야 (22~24시)
        return 0.15


def calculate_realtime_danger(edge_data: dict, hour: int = None) -> float:
    """
    실시간 시간대를 반영한 위험도 계산
    
    Args:
        edge_data: 엣지의 기본 데이터 (시설물 정보 등)
        hour: 시간 (None이면 현재 시간)
    
    Returns:
        위험도 (0~1)
    """
    if hour is None:
        hour = datetime.now().hour
    
    # 기본 위험도 (그래프에 저장된 값 또는 기본값)
    # 점수가 너무 짜게 나오는 경향이 있어 0.8을 곱해 전체적으로 상향 조정 (보정)
    base_danger = edge_data.get('danger_score', 0.5) * 0.8
    
    # 시간대 조정
    time_adjust = get_time_danger_adjustment(hour)
    
    # 가로등 효과 조정
    streetlight_count = edge_data.get('streetlight_count', 0)
    if is_streetlight_on(hour):
        # 야간에 가로등 있으면 안전
        streetlight_effect = -min(streetlight_count * 0.02, 0.1)
    else:
        # 주간에는 가로등 효과 없음
        streetlight_effect = 0
    
    # ★ 도로 유형 조정 (대로변 우선)
    highway_type = edge_data.get('highway', 'unknown')
    if isinstance(highway_type, list):
        highway_type = highway_type[0] if highway_type else 'unknown'
    highway_type = str(highway_type).lower()
    
    # 대로변 - 크게 안전 보너스 (경로 우선 선택되도록)
    if highway_type in ['trunk', 'trunk_link', 'primary', 'primary_link']:
        highway_adjust = -0.30  # 대로 - 매우 안전
    elif highway_type in ['secondary', 'secondary_link']:
        highway_adjust = -0.25  # 중로 - 안전
    elif highway_type in ['tertiary', 'tertiary_link']:
        highway_adjust = -0.20  # 소로 - 비교적 안전
    elif highway_type in ['residential', 'unclassified', 'living_street']:
        highway_adjust = 0.0   # 주거지 도로 - 보통
    elif highway_type in ['service', 'alley']:
        highway_adjust = 0.30  # 골목 - 페널티 강화 (0.20 → 0.30)
    elif highway_type in ['path', 'footway', 'pedestrian', 'steps', 'corridor']:
        highway_adjust = 0.40  # 보행자 통로/건물 내 통로 - 강력 페널티 (0.15 → 0.40)
    elif highway_type in ['cycleway', 'bridleway', 'track']:
        highway_adjust = 0.25  # 자전거/비포장 도로 - 중간 페널티
    else:
        highway_adjust = 0.35  # 알 수 없음 - 강력 페널티 (0.10 → 0.35)
    
    # 최종 위험도
    final_danger = base_danger + time_adjust + streetlight_effect + highway_adjust
    
    return np.clip(final_danger, 0.1, 0.9)


def calculate_realtime_weight(edge_data: dict, hour: int = None) -> float:
    """실시간 안전 가중치 계산 (대로변 우선)"""
    length = edge_data.get('length', 100)
    danger = calculate_realtime_danger(edge_data, hour)
    # 위험도 배수 강화: 2 → 3 (골목 페널티 증가, 대로변 선호)
    return length * (1 + danger * 3)


def find_nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> Optional[int]:
    """좌표에서 가장 가까운 노드 찾기"""
    if not LIBS_AVAILABLE:
        raise ImportError("osmnx 패키지 필요")
    
    try:
        node = ox.nearest_nodes(G, lon, lat)
        return node
    except:
        return None


def find_shortest_path(G: nx.MultiDiGraph, 
                       origin: Tuple[float, float], 
                       destination: Tuple[float, float],
                       hour: int = None) -> Dict:
    """최단 경로 탐색 (거리 기준)"""
    if hour is None:
        hour = datetime.now().hour
    
    orig_node = find_nearest_node(G, origin[0], origin[1])
    dest_node = find_nearest_node(G, destination[0], destination[1])
    
    if orig_node is None or dest_node is None:
        return {'error': '출발지 또는 목적지를 찾을 수 없습니다.'}
    
    try:
        path = nx.shortest_path(G, orig_node, dest_node, weight='length')
        
        total_length = 0
        danger_scores = []
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                first_edge = list(edge_data.values())[0]
                total_length += first_edge.get('length', 0)
                # 실시간 위험도 계산
                danger = calculate_realtime_danger(first_edge, hour)
                danger_scores.append(danger)
        
        avg_danger = np.mean(danger_scores) if danger_scores else 0.5
        
        return {
            'path': path,
            'length': float(total_length),
            'avg_danger_score': float(avg_danger),
            'avg_safety_score': int(100 * (1 - avg_danger)),
            'min_safety_score': int(100 * (1 - max(danger_scores))) if danger_scores else 50,
            'hour': hour,
            'type': 'shortest'
        }
    except nx.NetworkXNoPath:
        return {'error': '경로를 찾을 수 없습니다.'}


def find_safest_path(G: nx.MultiDiGraph, 
                     origin: Tuple[float, float], 
                     destination: Tuple[float, float],
                     hour: int = None) -> Dict:
    """최안전 경로 탐색 (실시간 안전 가중치 기준)"""
    if hour is None:
        hour = datetime.now().hour
    
    orig_node = find_nearest_node(G, origin[0], origin[1])
    dest_node = find_nearest_node(G, destination[0], destination[1])
    
    if orig_node is None or dest_node is None:
        return {'error': '출발지 또는 목적지를 찾을 수 없습니다.'}
    
    try:
        # 실시간 가중치 함수 (MultiDiGraph는 u, v, {key: data})
        def realtime_weight(u, v, edge_dict):
            # 병렬 엣지(Parallel Edges) 중 가장 안전한(가중치가 낮은) 엣지 선택
            weights = [calculate_realtime_weight(data, hour) for data in edge_dict.values()]
            return min(weights) if weights else float('inf')
        
        path = nx.shortest_path(G, orig_node, dest_node, weight=realtime_weight)
        
        total_length = 0
        danger_scores = []
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                first_edge = list(edge_data.values())[0]
                total_length += first_edge.get('length', 0)
                danger = calculate_realtime_danger(first_edge, hour)
                danger_scores.append(danger)
        
        avg_danger = np.mean(danger_scores) if danger_scores else 0.5
        
        return {
            'path': path,
            'length': float(total_length),
            'avg_danger_score': float(avg_danger),
            'avg_safety_score': int(100 * (1 - avg_danger)),
            'min_safety_score': int(100 * (1 - max(danger_scores))) if danger_scores else 50,
            'hour': hour,
            'type': 'safest'
        }
    except nx.NetworkXNoPath:
        return {'error': '경로를 찾을 수 없습니다.'}


def compare_routes(G: nx.MultiDiGraph, 
                   origin: Tuple[float, float], 
                   destination: Tuple[float, float],
                   hour: int = None) -> Dict:
    """최단 경로와 최안전 경로 비교"""
    if hour is None:
        hour = datetime.now().hour
    
    shortest = find_shortest_path(G, origin, destination, hour)
    safest = find_safest_path(G, origin, destination, hour)
    
    if 'error' in shortest or 'error' in safest:
        return {'error': shortest.get('error') or safest.get('error')}
    
    length_diff = safest['length'] - shortest['length']
    length_diff_pct = (length_diff / shortest['length'] * 100) if shortest['length'] > 0 else 0
    
    return {
        'shortest': shortest,
        'safest': safest,
        'length_difference': float(length_diff),
        'length_difference_percent': float(length_diff_pct),
        'safety_improvement': int(safest['avg_safety_score'] - shortest['avg_safety_score']),
        'current_hour': hour,
        'streetlight_on': is_streetlight_on(hour)
    }


def get_path_coords(G: nx.MultiDiGraph, path: List) -> List[Tuple[float, float]]:
    """경로 노드 → 좌표 리스트 변환"""
    return [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]


def search_route(start_lat: float, start_lon: float,
                 end_lat: float, end_lon: float,
                 hour: int = None) -> Dict:
    """안전 경로 검색 메인 함수"""
    if hour is None:
        hour = datetime.now().hour
    
    print("=" * 60)
    print("🧭 안전 경로 검색 (실시간)")
    print("=" * 60)
    
    print(f"\n⏰ 현재 시간: {hour}시 (가로등: {'ON' if is_streetlight_on(hour) else 'OFF'})")
    
    print("\n📂 그래프 로드...")
    try:
        G = load_graph()
        print(f"   ✅ 노드: {G.number_of_nodes():,}, 엣지: {G.number_of_edges():,}")
    except FileNotFoundError:
        print("❌ 그래프 파일이 없습니다.")
        return {'error': '그래프 파일 없음'}
    
    origin = (start_lat, start_lon)
    destination = (end_lat, end_lon)
    
    print(f"\n📍 출발: ({start_lat:.4f}, {start_lon:.4f})")
    print(f"📍 도착: ({end_lat:.4f}, {end_lon:.4f})")
    
    print("\n🔍 경로 탐색 중...")
    comparison = compare_routes(G, origin, destination, hour)
    
    if 'error' in comparison:
        print(f"❌ {comparison['error']}")
        return comparison
    
    print(f"\n📊 결과 ({hour}시 기준):")
    print(f"   🔵 최단: {comparison['shortest']['length']:.0f}m (안전: {comparison['shortest']['avg_safety_score']}점)")
    print(f"   🟢 안전: {comparison['safest']['length']:.0f}m (안전: {comparison['safest']['avg_safety_score']}점)")
    print(f"   📈 차이: {comparison['length_difference']:+.0f}m, 안전 {comparison['safety_improvement']:+}점")
    
    return comparison


def main():
    """테스트 - 시간대별 비교"""
    print("\n🕐 시간대별 안전 점수 비교 (서울역→시청)")
    print("-" * 50)
    
    for hour in [6, 12, 18, 22, 2]:
        result = search_route(37.5546, 126.9706, 37.5665, 126.9780, hour=hour)
        if 'error' not in result:
            sl = "🌙" if is_streetlight_on(hour) else "☀️"
            print(f"{hour:02d}시 {sl}: 안전 {result['safest']['avg_safety_score']}점")


if __name__ == "__main__":
    main()
