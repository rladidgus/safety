"""
안전 경로 탐색 모듈
- 최단 경로 vs 최안전 경로 비교
- 경로 시각화
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import osmnx as ox
    import networkx as nx
    import folium
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False

# graph_builder에서 필요한 것만 가져오기 (스크립트/모듈 모두 지원)
try:
    from .graph_builder import load_graph, MODEL_DIR
except ImportError:
    try:
        from graph_builder import load_graph, MODEL_DIR
    except ImportError:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from graph_builder import load_graph, MODEL_DIR


def find_nearest_node(G, lat: float, lon: float) -> int:
    """주어진 좌표에서 가장 가까운 노드 찾기"""
    return ox.nearest_nodes(G, lon, lat)


def find_shortest_path(G, origin: Tuple[float, float], 
                       destination: Tuple[float, float]) -> dict:
    """
    최단 경로 탐색 (거리 기준)
    
    Args:
        G: 그래프
        origin: 출발지 (lat, lon)
        destination: 목적지 (lat, lon)
    
    Returns:
        경로 정보 딕셔너리
    """
    orig_node = find_nearest_node(G, origin[0], origin[1])
    dest_node = find_nearest_node(G, destination[0], destination[1])
    
    try:
        path = nx.shortest_path(G, orig_node, dest_node, weight='length')
        length = nx.shortest_path_length(G, orig_node, dest_node, weight='length')
        
        # 경로의 안전 점수 계산
        safety_scores = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = G.edges[u, v, 0] if G.has_edge(u, v) else {}
            safety_scores.append(edge_data.get('safety_score', 50))
        
        return {
            'path': path,
            'length': length,
            'avg_safety_score': np.mean(safety_scores) if safety_scores else 50,
            'min_safety_score': np.min(safety_scores) if safety_scores else 50,
            'type': 'shortest'
        }
    except nx.NetworkXNoPath:
        return {'error': '경로를 찾을 수 없습니다.'}


def find_safest_path(G, origin: Tuple[float, float], 
                     destination: Tuple[float, float]) -> dict:
    """
    최안전 경로 탐색 (안전 가중치 기준)
    
    Args:
        G: 그래프
        origin: 출발지 (lat, lon)
        destination: 목적지 (lat, lon)
    
    Returns:
        경로 정보 딕셔너리
    """
    orig_node = find_nearest_node(G, origin[0], origin[1])
    dest_node = find_nearest_node(G, destination[0], destination[1])
    
    try:
        path = nx.shortest_path(G, orig_node, dest_node, weight='safety_weight')
        
        # 실제 거리 계산
        length = 0
        safety_scores = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = G.edges[u, v, 0] if G.has_edge(u, v) else {}
            length += edge_data.get('length', 0)
            safety_scores.append(edge_data.get('safety_score', 50))
        
        return {
            'path': path,
            'length': length,
            'avg_safety_score': np.mean(safety_scores) if safety_scores else 50,
            'min_safety_score': np.min(safety_scores) if safety_scores else 50,
            'type': 'safest'
        }
    except nx.NetworkXNoPath:
        return {'error': '경로를 찾을 수 없습니다.'}


def compare_routes(G, origin: Tuple[float, float], 
                   destination: Tuple[float, float]) -> dict:
    """
    최단 경로와 최안전 경로 비교
    
    Args:
        G: 그래프
        origin: 출발지 (lat, lon)
        destination: 목적지 (lat, lon)
    
    Returns:
        비교 결과
    """
    shortest = find_shortest_path(G, origin, destination)
    safest = find_safest_path(G, origin, destination)
    
    if 'error' in shortest or 'error' in safest:
        return {'error': '경로를 찾을 수 없습니다.'}
    
    return {
        'shortest': shortest,
        'safest': safest,
        'length_difference': safest['length'] - shortest['length'],
        'length_difference_percent': (safest['length'] - shortest['length']) / shortest['length'] * 100 if shortest['length'] > 0 else 0,
        'safety_improvement': safest['avg_safety_score'] - shortest['avg_safety_score']
    }


def visualize_route(G, route_info: dict, 
                    filename: str = "route_map.html") -> str:
    """
    경로를 지도에 시각화
    
    Args:
        G: 그래프
        route_info: 경로 정보 (find_*_path 반환값)
        filename: 저장할 파일명
    
    Returns:
        저장된 파일 경로
    """
    if 'error' in route_info or 'path' not in route_info:
        print("❌ 시각화할 경로가 없습니다.")
        return ""
    
    path = route_info['path']
    
    # 경로 좌표 추출
    coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in path]
    
    # 지도 중심점
    center_lat = np.mean([c[0] for c in coords])
    center_lon = np.mean([c[1] for c in coords])
    
    # Folium 지도 생성
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
    
    # 경로 색상 (최단: 파랑, 최안전: 초록)
    color = 'green' if route_info.get('type') == 'safest' else 'blue'
    
    # 경로 그리기
    folium.PolyLine(
        coords,
        weight=5,
        color=color,
        opacity=0.8
    ).add_to(m)
    
    # 출발점/도착점 마커
    folium.Marker(
        coords[0],
        popup='출발',
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    folium.Marker(
        coords[-1],
        popup='도착',
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)
    
    # 저장
    output_path = MODEL_DIR / filename
    m.save(str(output_path))
    print(f"✅ 지도 저장: {output_path}")
    
    return str(output_path)


def visualize_comparison(G, comparison: dict, 
                         filename: str = "route_comparison.html") -> str:
    """
    최단 경로와 최안전 경로 비교 시각화
    
    Args:
        G: 그래프
        comparison: compare_routes 반환값
        filename: 저장할 파일명
    
    Returns:
        저장된 파일 경로
    """
    if 'error' in comparison:
        print("❌ 시각화할 경로가 없습니다.")
        return ""
    
    shortest_path = comparison['shortest']['path']
    safest_path = comparison['safest']['path']
    
    # 좌표 추출
    shortest_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in shortest_path]
    safest_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in safest_path]
    
    # 지도 중심점
    all_coords = shortest_coords + safest_coords
    center_lat = np.mean([c[0] for c in all_coords])
    center_lon = np.mean([c[1] for c in all_coords])
    
    # Folium 지도 생성
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
    
    # 최단 경로 (파랑)
    folium.PolyLine(
        shortest_coords,
        weight=5,
        color='blue',
        opacity=0.7,
        popup=f"최단 경로: {comparison['shortest']['length']:.0f}m, 안전점수: {comparison['shortest']['avg_safety_score']:.1f}"
    ).add_to(m)
    
    # 최안전 경로 (초록)
    folium.PolyLine(
        safest_coords,
        weight=5,
        color='green',
        opacity=0.7,
        popup=f"안전 경로: {comparison['safest']['length']:.0f}m, 안전점수: {comparison['safest']['avg_safety_score']:.1f}"
    ).add_to(m)
    
    # 출발점/도착점 마커
    folium.Marker(
        shortest_coords[0],
        popup='출발',
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    folium.Marker(
        shortest_coords[-1],
        popup='도착',
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)
    
    # 범례 추가
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background: white; padding: 10px; border-radius: 5px; box-shadow: 2px 2px 5px grey;">
        <p><strong>경로 비교</strong></p>
        <p><span style="color: blue;">━━</span> 최단 경로</p>
        <p><span style="color: green;">━━</span> 안전 경로</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # 저장
    output_path = MODEL_DIR / filename
    m.save(str(output_path))
    print(f"✅ 비교 지도 저장: {output_path}")
    
    return str(output_path)


def main():
    """테스트 실행"""
    print("=" * 60)
    print("🧭 안전 경로 탐색 테스트")
    print("=" * 60)
    
    try:
        # 그래프 로드
        print("\n📂 그래프 로드 중...")
        G = load_graph("safety_graph")
        print(f"   ✅ 노드: {G.number_of_nodes():,}, 엣지: {G.number_of_edges():,}")
        
        # 테스트 좌표 (강남역 → 삼성역)
        origin = (37.4979, 127.0276)      # 강남역
        destination = (37.5089, 127.0631)  # 삼성역
        
        print(f"\n📍 출발: {origin}")
        print(f"📍 도착: {destination}")
        
        # 경로 비교
        print("\n🔍 경로 탐색 중...")
        comparison = compare_routes(G, origin, destination)
        
        if 'error' not in comparison:
            print(f"\n📊 결과:")
            print(f"   최단 경로: {comparison['shortest']['length']:.0f}m (안전점수: {comparison['shortest']['avg_safety_score']:.1f})")
            print(f"   안전 경로: {comparison['safest']['length']:.0f}m (안전점수: {comparison['safest']['avg_safety_score']:.1f})")
            print(f"   거리 차이: +{comparison['length_difference']:.0f}m ({comparison['length_difference_percent']:.1f}%)")
            print(f"   안전 향상: +{comparison['safety_improvement']:.1f}점")
            
            # 시각화
            visualize_comparison(G, comparison)
        else:
            print(f"❌ {comparison['error']}")
            
    except FileNotFoundError:
        print("❌ 그래프 파일이 없습니다. 먼저 graph_builder.py를 실행해주세요.")


if __name__ == "__main__":
    main()
