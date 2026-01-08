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

# graph_builder에서 필요한 것만 가져오기
try:
    from .graph_builder import load_graph, MODEL_DIR
except ImportError:
    from graph_builder import load_graph, MODEL_DIR


def find_nearest_node(G, lat: float, lon: float) -> int:
    """주어진 좌표에서 가장 가까운 노드 찾기"""
    return ox.nearest_nodes(G, lon, lat)


def find_shortest_path(G, origin: Tuple[float, float], 
                       destination: Tuple[float, float]) -> dict:
    """
    최단 경로 탐색 (거리 기준)
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
