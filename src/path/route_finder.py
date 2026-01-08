"""
안전 경로 탐색 모듈
- A* 알고리즘 기반 경로 탐색
- 최단 경로 vs 최안전 경로 비교
- Folium 지도 시각화
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    import networkx as nx
    from scipy.spatial import cKDTree
    LIBS_AVAILABLE = True
except ImportError as e:
    LIBS_AVAILABLE = False
    print(f"⚠️ 필요한 패키지: pip install networkx scipy")

try:
    import folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 좌표 간 거리 계산 (미터)"""
    R = 6371000
    
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    
    a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c


def load_graph(filename: str = "pedestrian_safety_graph") -> nx.Graph:
    """저장된 그래프 로드"""
    filepath = MODEL_DIR / f"{filename}.pkl"
    
    if not filepath.exists():
        raise FileNotFoundError(f"그래프 파일 없음: {filepath}")
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def find_nearest_node(G: nx.Graph, lat: float, lon: float) -> Optional[int]:
    """좌표에서 가장 가까운 노드 찾기"""
    if not LIBS_AVAILABLE:
        raise ImportError("scipy 패키지 필요")
    
    nodes = list(G.nodes())
    coords = np.array([(G.nodes[n].get('lat', 0), G.nodes[n].get('lon', 0)) for n in nodes])
    
    tree = cKDTree(coords)
    dist, idx = tree.query([lat, lon])
    
    if dist * 111000 > 1000:
        return None
    
    return nodes[idx]


def find_shortest_path(G: nx.Graph, 
                       origin: Tuple[float, float], 
                       destination: Tuple[float, float]) -> Dict:
    """최단 경로 탐색 (거리 기준)"""
    orig_node = find_nearest_node(G, origin[0], origin[1])
    dest_node = find_nearest_node(G, destination[0], destination[1])
    
    if orig_node is None or dest_node is None:
        return {'error': '출발지 또는 목적지가 도로망 범위를 벗어났습니다.'}
    
    try:
        path = nx.shortest_path(G, orig_node, dest_node, weight='length')
        
        total_length = 0
        danger_scores = []
        
        for i in range(len(path) - 1):
            edge_data = G.edges[path[i], path[i+1]]
            total_length += edge_data.get('length', 0)
            danger_scores.append(edge_data.get('danger_score', 0.5))
        
        avg_danger = np.mean(danger_scores) if danger_scores else 0.5
        
        return {
            'path': path,
            'length': total_length,
            'avg_danger_score': avg_danger,
            'avg_safety_score': int(100 * (1 - avg_danger)),
            'min_safety_score': int(100 * (1 - max(danger_scores))) if danger_scores else 50,
            'type': 'shortest'
        }
    except nx.NetworkXNoPath:
        return {'error': '경로를 찾을 수 없습니다.'}


def find_safest_path(G: nx.Graph, 
                     origin: Tuple[float, float], 
                     destination: Tuple[float, float]) -> Dict:
    """최안전 경로 탐색 (위험도 가중치 기준)"""
    orig_node = find_nearest_node(G, origin[0], origin[1])
    dest_node = find_nearest_node(G, destination[0], destination[1])
    
    if orig_node is None or dest_node is None:
        return {'error': '출발지 또는 목적지가 도로망 범위를 벗어났습니다.'}
    
    try:
        path = nx.shortest_path(G, orig_node, dest_node, weight='weight')
        
        total_length = 0
        danger_scores = []
        
        for i in range(len(path) - 1):
            edge_data = G.edges[path[i], path[i+1]]
            total_length += edge_data.get('length', 0)
            danger_scores.append(edge_data.get('danger_score', 0.5))
        
        avg_danger = np.mean(danger_scores) if danger_scores else 0.5
        
        return {
            'path': path,
            'length': total_length,
            'avg_danger_score': avg_danger,
            'avg_safety_score': int(100 * (1 - avg_danger)),
            'min_safety_score': int(100 * (1 - max(danger_scores))) if danger_scores else 50,
            'type': 'safest'
        }
    except nx.NetworkXNoPath:
        return {'error': '경로를 찾을 수 없습니다.'}


def compare_routes(G: nx.Graph, 
                   origin: Tuple[float, float], 
                   destination: Tuple[float, float]) -> Dict:
    """최단 경로와 최안전 경로 비교"""
    shortest = find_shortest_path(G, origin, destination)
    safest = find_safest_path(G, origin, destination)
    
    if 'error' in shortest or 'error' in safest:
        return {'error': shortest.get('error') or safest.get('error')}
    
    length_diff = safest['length'] - shortest['length']
    length_diff_pct = (length_diff / shortest['length'] * 100) if shortest['length'] > 0 else 0
    
    return {
        'shortest': shortest,
        'safest': safest,
        'length_difference': length_diff,
        'length_difference_percent': length_diff_pct,
        'safety_improvement': safest['avg_safety_score'] - shortest['avg_safety_score']
    }


def get_path_coords(G: nx.Graph, path: List) -> List[Tuple[float, float]]:
    """경로 노드 → 좌표 리스트 변환"""
    return [(G.nodes[n].get('lat', 0), G.nodes[n].get('lon', 0)) for n in path]


def visualize_comparison(G: nx.Graph, comparison: Dict, 
                         filename: str = "route_comparison.html") -> str:
    """최단 경로와 최안전 경로 비교 시각화"""
    if not FOLIUM_AVAILABLE:
        print("⚠️ folium 패키지가 필요합니다: pip install folium")
        return ""
    
    if 'error' in comparison:
        print(f"❌ 시각화 실패: {comparison['error']}")
        return ""
    
    shortest_coords = get_path_coords(G, comparison['shortest']['path'])
    safest_coords = get_path_coords(G, comparison['safest']['path'])
    
    all_coords = shortest_coords + safest_coords
    center_lat = np.mean([c[0] for c in all_coords])
    center_lon = np.mean([c[1] for c in all_coords])
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
    
    folium.PolyLine(
        shortest_coords, weight=4, color='blue', opacity=0.7, dash_array='10',
        popup=f"최단: {comparison['shortest']['length']:.0f}m"
    ).add_to(m)
    
    folium.PolyLine(
        safest_coords, weight=5, color='green', opacity=0.8,
        popup=f"안전: {comparison['safest']['length']:.0f}m"
    ).add_to(m)
    
    folium.Marker(shortest_coords[0], popup='출발', 
                  icon=folium.Icon(color='green', icon='play')).add_to(m)
    folium.Marker(shortest_coords[-1], popup='도착', 
                  icon=folium.Icon(color='red', icon='stop')).add_to(m)
    
    MODEL_DIR.mkdir(exist_ok=True)
    output_path = MODEL_DIR / filename
    m.save(str(output_path))
    print(f"✅ 지도 저장: {output_path}")
    
    return str(output_path)


def search_route(start_lat: float, start_lon: float,
                 end_lat: float, end_lon: float,
                 visualize: bool = True) -> Dict:
    """안전 경로 검색 메인 함수"""
    print("=" * 60)
    print("🧭 안전 경로 검색")
    print("=" * 60)
    
    print("\n📂 그래프 로드...")
    try:
        G = load_graph()
        print(f"   ✅ 노드: {G.number_of_nodes():,}, 엣지: {G.number_of_edges():,}")
    except FileNotFoundError:
        print("❌ 그래프 파일이 없습니다.")
        print("   python src/graph_builder.py 를 먼저 실행하세요.")
        return {'error': '그래프 파일 없음'}
    
    origin = (start_lat, start_lon)
    destination = (end_lat, end_lon)
    
    print(f"\n📍 출발: ({start_lat:.4f}, {start_lon:.4f})")
    print(f"📍 도착: ({end_lat:.4f}, {end_lon:.4f})")
    
    direct_dist = haversine_distance(start_lat, start_lon, end_lat, end_lon)
    print(f"📏 직선 거리: {direct_dist:.0f}m")
    
    print("\n🔍 경로 탐색 중...")
    comparison = compare_routes(G, origin, destination)
    
    if 'error' in comparison:
        print(f"❌ {comparison['error']}")
        return comparison
    
    print(f"\n{'='*40}")
    print(f"📊 결과")
    print(f"{'='*40}")
    print(f"   🔵 최단 경로: {comparison['shortest']['length']:.0f}m (안전: {comparison['shortest']['avg_safety_score']}점)")
    print(f"   🟢 안전 경로: {comparison['safest']['length']:.0f}m (안전: {comparison['safest']['avg_safety_score']}점)")
    print(f"   📈 거리 차이: {comparison['length_difference']:+.0f}m ({comparison['length_difference_percent']:+.1f}%)")
    print(f"   📈 안전 향상: {comparison['safety_improvement']:+}점")
    
    if visualize:
        print("\n🗺️ 지도 생성...")
        visualize_comparison(G, comparison)
    
    print("\n" + "=" * 60)
    print("✅ 검색 완료!")
    print("=" * 60)
    
    return comparison


def main():
    """테스트 실행"""
    try:
        G = load_graph()
        nodes = list(G.nodes())
        
        if len(nodes) >= 2:
            start_node = nodes[0]
            end_node = nodes[min(100, len(nodes)-1)]
            
            start_lat = G.nodes[start_node].get('lat', 37.5)
            start_lon = G.nodes[start_node].get('lon', 127.0)
            end_lat = G.nodes[end_node].get('lat', 37.5)
            end_lon = G.nodes[end_node].get('lon', 127.0)
            
            search_route(start_lat, start_lon, end_lat, end_lon)
            
    except FileNotFoundError:
        print("❌ 그래프 파일이 없습니다.")
        print("   python src/graph_builder.py 를 먼저 실행하세요.")


if __name__ == "__main__":
    main()
