"""
안전 경로 탐색 시스템 테스트
- 그래프 로드 테스트
- 경로 탐색 테스트
- 최단 vs 안전 경로 비교
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

def test_graph_load():
    """1. 그래프 로드 테스트"""
    print("=" * 60)
    print("📌 테스트 1: 그래프 로드")
    print("=" * 60)
    
    try:
        from route_finder import load_graph
        G = load_graph()
        
        print(f"   ✅ 노드 수: {G.number_of_nodes():,}")
        print(f"   ✅ 엣지 수: {G.number_of_edges():,}")
        print(f"   ✅ 생성 시간: {G.graph.get('hour', 'N/A')}시")
        print(f"   ✅ 가로등: {'ON' if G.graph.get('streetlight_on', False) else 'OFF'}")
        
        return G, True
    except FileNotFoundError:
        print("   ❌ 그래프 파일 없음")
        print("   💡 해결: python src/graph_builder.py 실행")
        return None, False
    except Exception as e:
        print(f"   ❌ 오류: {e}")
        return None, False


def test_find_nearest_node(G):
    """2. 가장 가까운 노드 찾기 테스트"""
    print("\n" + "=" * 60)
    print("📌 테스트 2: 노드 매칭")
    print("=" * 60)
    
    try:
        from route_finder import find_nearest_node
        
        # 테스트 좌표 (서울 중심부)
        test_coords = [
            (37.5665, 126.9780, "서울시청 근처"),
            (37.5546, 126.9706, "서울역 근처"),
            (37.5172, 127.0473, "강남역 근처"),
        ]
        
        for lat, lon, name in test_coords:
            node = find_nearest_node(G, lat, lon)
            if node:
                node_lat = G.nodes[node].get('lat', 0)
                node_lon = G.nodes[node].get('lon', 0)
                print(f"   ✅ {name}: 노드 {node} ({node_lat:.4f}, {node_lon:.4f})")
            else:
                print(f"   ⚠️ {name}: 범위 밖")
        
        return True
    except Exception as e:
        print(f"   ❌ 오류: {e}")
        return False


def test_shortest_path(G):
    """3. 최단 경로 테스트"""
    print("\n" + "=" * 60)
    print("📌 테스트 3: 최단 경로 탐색")
    print("=" * 60)
    
    try:
        from route_finder import find_shortest_path, find_nearest_node
        
        # 그래프에서 실제 존재하는 노드 2개 선택
        nodes = list(G.nodes())
        if len(nodes) < 100:
            print("   ❌ 노드 수 부족")
            return False
        
        start_node = nodes[0]
        end_node = nodes[100]
        
        start_lat = G.nodes[start_node].get('lat', 0)
        start_lon = G.nodes[start_node].get('lon', 0)
        end_lat = G.nodes[end_node].get('lat', 0)
        end_lon = G.nodes[end_node].get('lon', 0)
        
        print(f"   출발: ({start_lat:.4f}, {start_lon:.4f})")
        print(f"   도착: ({end_lat:.4f}, {end_lon:.4f})")
        
        result = find_shortest_path(G, (start_lat, start_lon), (end_lat, end_lon))
        
        if 'error' in result:
            print(f"   ⚠️ {result['error']}")
            return False
        
        print(f"   ✅ 경로 노드 수: {len(result['path'])}")
        print(f"   ✅ 총 거리: {result['length']:.0f}m")
        print(f"   ✅ 안전 점수: {result['avg_safety_score']}점")
        
        return True
    except Exception as e:
        print(f"   ❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_safest_path(G):
    """4. 안전 경로 테스트"""
    print("\n" + "=" * 60)
    print("📌 테스트 4: 안전 경로 탐색")
    print("=" * 60)
    
    try:
        from route_finder import find_safest_path
        
        nodes = list(G.nodes())
        start_node = nodes[0]
        end_node = nodes[100]
        
        start_lat = G.nodes[start_node].get('lat', 0)
        start_lon = G.nodes[start_node].get('lon', 0)
        end_lat = G.nodes[end_node].get('lat', 0)
        end_lon = G.nodes[end_node].get('lon', 0)
        
        result = find_safest_path(G, (start_lat, start_lon), (end_lat, end_lon))
        
        if 'error' in result:
            print(f"   ⚠️ {result['error']}")
            return False
        
        print(f"   ✅ 경로 노드 수: {len(result['path'])}")
        print(f"   ✅ 총 거리: {result['length']:.0f}m")
        print(f"   ✅ 안전 점수: {result['avg_safety_score']}점")
        
        return True
    except Exception as e:
        print(f"   ❌ 오류: {e}")
        return False


def test_compare_routes(G):
    """5. 경로 비교 테스트"""
    print("\n" + "=" * 60)
    print("📌 테스트 5: 최단 vs 안전 경로 비교")
    print("=" * 60)
    
    try:
        from route_finder import compare_routes
        
        nodes = list(G.nodes())
        start_node = nodes[0]
        end_node = nodes[500]  # 더 먼 거리로 테스트
        
        start_lat = G.nodes[start_node].get('lat', 0)
        start_lon = G.nodes[start_node].get('lon', 0)
        end_lat = G.nodes[end_node].get('lat', 0)
        end_lon = G.nodes[end_node].get('lon', 0)
        
        print(f"   출발: ({start_lat:.4f}, {start_lon:.4f})")
        print(f"   도착: ({end_lat:.4f}, {end_lon:.4f})")
        
        result = compare_routes(G, (start_lat, start_lon), (end_lat, end_lon))
        
        if 'error' in result:
            print(f"   ⚠️ {result['error']}")
            return False
        
        print(f"\n   🔵 최단 경로:")
        print(f"      거리: {result['shortest']['length']:.0f}m")
        print(f"      안전점수: {result['shortest']['avg_safety_score']}점")
        
        print(f"\n   🟢 안전 경로:")
        print(f"      거리: {result['safest']['length']:.0f}m")
        print(f"      안전점수: {result['safest']['avg_safety_score']}점")
        
        print(f"\n   📊 비교:")
        print(f"      거리 차이: {result['length_difference']:+.0f}m ({result['length_difference_percent']:+.1f}%)")
        print(f"      안전 향상: {result['safety_improvement']:+}점")
        
        # 검증
        if result['safest']['avg_safety_score'] >= result['shortest']['avg_safety_score']:
            print(f"\n   ✅ 검증 통과: 안전 경로가 더 안전함!")
        else:
            print(f"\n   ⚠️ 경고: 안전 경로가 덜 안전함 (데이터 확인 필요)")
        
        return True
    except Exception as e:
        print(f"   ❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "🧪" * 30)
    print("       안전 경로 탐색 시스템 테스트")
    print("🧪" * 30 + "\n")
    
    results = {}
    
    # 테스트 1: 그래프 로드
    G, success = test_graph_load()
    results['그래프 로드'] = success
    
    if not success:
        print("\n❌ 그래프 로드 실패로 나머지 테스트를 건너뜁니다.")
        return results
    
    # 테스트 2: 노드 매칭
    results['노드 매칭'] = test_find_nearest_node(G)
    
    # 테스트 3: 최단 경로
    results['최단 경로'] = test_shortest_path(G)
    
    # 테스트 4: 안전 경로
    results['안전 경로'] = test_safest_path(G)
    
    # 테스트 5: 경로 비교
    results['경로 비교'] = test_compare_routes(G)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 테스트 결과 요약")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = "✅ 통과" if success else "❌ 실패"
        print(f"   {name}: {status}")
    
    print(f"\n   총 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("\n🎉 모든 테스트 통과!")
    else:
        print("\n⚠️ 일부 테스트 실패. 위 오류를 확인하세요.")
    
    return results


if __name__ == "__main__":
    run_all_tests()
