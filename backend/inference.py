"""
추론 모듈
- 학습된 모델을 이용한 안전 점수 예측
- 경로의 안전성 평가
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import joblib


# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


class SafetyPredictor:
    """안전 점수 예측기"""
    
    def __init__(self, model_name: str = "safety_model"):
        """
        모델 로드
        
        Args:
            model_name: 모델 파일 이름 (확장자 제외)
        """
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._load_model()
    
    def _load_model(self):
        """저장된 모델, 스케일러, 메타데이터 로드"""
        model_path = MODEL_DIR / f"{self.model_name}.joblib"
        scaler_path = MODEL_DIR / f"{self.model_name}_scaler.joblib"
        metadata_path = MODEL_DIR / f"{self.model_name}_metadata.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        metadata = joblib.load(metadata_path)
        self.feature_names = metadata["feature_names"]
        
        print(f"✅ 모델 로드 완료: {self.model_name}")
    
    def predict(self, features: dict) -> float:
        """
        단일 지점의 안전 점수 예측
        
        Args:
            features: 피처 딕셔너리
                - dist_to_streetlight: 가로등까지 거리 (m)
                - dist_to_police: 파출소까지 거리 (m)
                - dist_to_main_road: 대로변까지 거리 (m)
                - streetlight_count_100m: 100m 반경 가로등 수
        
        Returns:
            안전 점수 (0-100)
        """
        # 입력 검증
        missing = [f for f in self.feature_names if f not in features]
        if missing:
            raise ValueError(f"누락된 피처: {missing}")
        
        # 피처 배열 준비
        X = np.array([[features[f] for f in self.feature_names]])
        X_scaled = self.scaler.transform(X)
        
        # 예측
        score = self.model.predict(X_scaled)[0]
        return float(np.clip(score, 0, 100))
    
    def predict_batch(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        여러 지점의 안전 점수 일괄 예측
        
        Args:
            features_df: 피처 DataFrame
        
        Returns:
            안전 점수 배열
        """
        X = features_df[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        scores = self.model.predict(X_scaled)
        return np.clip(scores, 0, 100)
    
    def evaluate_route(self, route_points: List[dict]) -> dict:
        """
        경로 전체의 안전성 평가
        
        Args:
            route_points: 경로를 구성하는 점들의 피처 리스트
        
        Returns:
            경로 안전성 평가 결과
        """
        if not route_points:
            return {"error": "경로 포인트가 없습니다"}
        
        # 각 포인트의 안전 점수 계산
        scores = [self.predict(point) for point in route_points]
        
        return {
            "average_score": float(np.mean(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "std_score": float(np.std(scores)),
            "danger_points": sum(1 for s in scores if s < 50),  # 50점 미만 위험 구간
            "point_scores": scores
        }
    
    def get_safety_level(self, score: float) -> str:
        """안전 점수를 레벨로 변환"""
        if score >= 80:
            return "매우 안전"
        elif score >= 60:
            return "안전"
        elif score >= 40:
            return "주의"
        elif score >= 20:
            return "위험"
        else:
            return "매우 위험"


def main():
    """추론 예시"""
    print("=" * 50)
    print("🔍 안전 점수 예측 테스트")
    print("=" * 50)
    
    try:
        # 예측기 초기화
        predictor = SafetyPredictor()
        
        # 단일 지점 예측 예시
        sample_point = {
            "dist_to_streetlight": 25.0,
            "dist_to_police": 300.0,
            "dist_to_main_road": 50.0,
            "streetlight_count_100m": 4
        }
        
        score = predictor.predict(sample_point)
        level = predictor.get_safety_level(score)
        
        print(f"\n📍 샘플 지점 예측:")
        print(f"   - 입력: {sample_point}")
        print(f"   - 안전 점수: {score:.1f}")
        print(f"   - 안전 레벨: {level}")
        
        # 경로 평가 예시
        sample_route = [
            {"dist_to_streetlight": 10, "dist_to_police": 200, "dist_to_main_road": 30, "streetlight_count_100m": 5},
            {"dist_to_streetlight": 50, "dist_to_police": 500, "dist_to_main_road": 100, "streetlight_count_100m": 2},
            {"dist_to_streetlight": 100, "dist_to_police": 800, "dist_to_main_road": 200, "streetlight_count_100m": 0},
            {"dist_to_streetlight": 20, "dist_to_police": 150, "dist_to_main_road": 20, "streetlight_count_100m": 6},
        ]
        
        route_eval = predictor.evaluate_route(sample_route)
        
        print(f"\n🛣️ 경로 평가:")
        print(f"   - 평균 점수: {route_eval['average_score']:.1f}")
        print(f"   - 최저 점수: {route_eval['min_score']:.1f}")
        print(f"   - 최고 점수: {route_eval['max_score']:.1f}")
        print(f"   - 위험 구간 수: {route_eval['danger_points']}")
        
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 먼저 train.py를 실행하여 모델을 학습해주세요.")


if __name__ == "__main__":
    main()
