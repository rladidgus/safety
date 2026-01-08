"""
안심 길 안내 서비스 - 통합 테스트 스크립트
전체 파이프라인을 순서대로 실행하고 결과를 확인합니다.
"""

import sys
from pathlib import Path

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

def test_step(step_name: str, func):
    """테스트 스텝 실행"""
    print(f"\n{'='*60}")
    print(f"🧪 테스트: {step_name}")
    print('='*60)
    try:
        result = func()
        print(f"✅ {step_name} 성공!")
        return result
    except Exception as e:
        print(f"❌ {step_name} 실패: {e}")
        return None


def test_preprocessing():
    """전처리 테스트"""
    from src.preprocessing import preprocess_all_data
    return preprocess_all_data()


def test_ml_training():
    """ML 학습 테스트"""
    from src.ml_trainer import train_and_save_model
    return train_and_save_model()


def test_graph_building():
    """그래프 구축 테스트 (작은 지역으로)"""
    from src.graph_builder import main
    # 작은 지역으로 빠르게 테스트
    return main("Gangnam-gu, Seoul, South Korea", hour=23)


def test_route_finding():
    """경로 탐색 테스트"""
    from src.route_finder import main as route_main
    route_main()


def test_ml_prediction():
    """ML 예측 단위 테스트"""
    from src.ml_trainer import SafetyMLModel
    
    model = SafetyMLModel()
    model.load()
    
    print("\n🔍 ML 예측 테스트:")
    
    # 테스트 케이스들
    test_cases = [
        {"name": "안전한 길 (시설물 많음)", 
         "streetlight": 5, "cctv": 3, "convenience": 2, "entertainment": 0, "police": 1},
        {"name": "위험한 길 (시설물 없음)", 
         "streetlight": 0, "cctv": 0, "convenience": 0, "entertainment": 3, "police": 0},
        {"name": "보통 길", 
         "streetlight": 2, "cctv": 1, "convenience": 1, "entertainment": 1, "police": 0},
        {"name": "가로등 없는 길", 
         "streetlight": 0, "cctv": 1, "convenience": 0, "entertainment": 0, "police": 0},
    ]
    
    for case in test_cases:
        danger = model.predict_single(
            case["streetlight"],
            case["cctv"],
            case["convenience"],
            case["entertainment"],
            case["police"]
        )
        safety_score = 100 * (1 - danger)
        
        status = "🟢 안전" if safety_score > 70 else "🟡 주의" if safety_score > 40 else "🔴 위험"
        
        print(f"\n   {case['name']}:")
        print(f"      가로등:{case['streetlight']} CCTV:{case['cctv']} "
              f"편의점:{case['convenience']} 유흥업소:{case['entertainment']}")
        print(f"      → 위험도: {danger:.3f}, 안전점수: {safety_score:.1f} {status}")


def run_quick_test():
    """빠른 테스트 (ML 예측만)"""
    print("=" * 60)
    print("🚀 빠른 테스트 시작 (ML 예측)")
    print("=" * 60)
    
    test_ml_prediction()
    
    print("\n" + "=" * 60)
    print("✅ 빠른 테스트 완료!")
    print("=" * 60)


def run_full_test():
    """전체 테스트"""
    print("=" * 60)
    print("🚀 전체 파이프라인 테스트 시작")
    print("=" * 60)
    
    # 1. 전처리
    data = test_step("데이터 전처리", test_preprocessing)
    
    # 2. ML 학습
    model = test_step("ML 모델 학습", test_ml_training)
    
    # 3. 그래프 구축
    # graph = test_step("그래프 구축", test_graph_building)
    
    # 4. ML 예측 테스트
    test_step("ML 예측 테스트", test_ml_prediction)
    
    print("\n" + "=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)
    print("\n💡 그래프 구축 테스트는 시간이 오래 걸려서 제외됨")
    print("   python src/graph_builder.py 로 별도 실행하세요")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="안심 길 안내 서비스 테스트")
    parser.add_argument("--quick", action="store_true", help="빠른 테스트 (ML만)")
    parser.add_argument("--full", action="store_true", help="전체 테스트")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    elif args.full:
        run_full_test()
    else:
        print("사용법:")
        print("  python test_pipeline.py --quick  # ML 예측 테스트만")
        print("  python test_pipeline.py --full   # 전체 파이프라인 테스트")
        print("\n기본으로 전체 테스트를 실행합니다...")
        run_full_test()
