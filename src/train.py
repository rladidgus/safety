"""
모델 학습 모듈
- 안전 점수 예측 모델 학습
- 모델 저장 및 평가
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

# 모델 디렉토리 생성
MODEL_DIR.mkdir(exist_ok=True)


def load_training_data(filepath: str) -> pd.DataFrame:
    """학습 데이터 로드"""
    df = pd.read_csv(filepath)
    print(f"✅ 학습 데이터 로드: {len(df)} 행")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    피처와 타겟 분리
    
    Returns:
        X: 피처 DataFrame
        y: 타겟 Series
    """
    feature_columns = [
        "dist_to_streetlight",
        "dist_to_police",
        "dist_to_main_road",
        "streetlight_count_100m"
    ]
    
    target_column = "safety_score"
    
    # 필요한 컬럼 확인
    missing_cols = [col for col in feature_columns + [target_column] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"누락된 컬럼: {missing_cols}")
    
    X = df[feature_columns]
    y = df[target_column]
    
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, model_type: str = "random_forest") -> dict:
    """
    모델 학습
    
    Args:
        X: 피처 데이터
        y: 타겟 데이터
        model_type: 'random_forest' 또는 'gradient_boosting'
    
    Returns:
        학습 결과 딕셔너리 (모델, 스케일러, 메트릭)
    """
    print(f"\n🔄 {model_type} 모델 학습 시작...")
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 선택 및 학습
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    else:
        raise ValueError(f"지원하지 않는 모델 타입: {model_type}")
    
    model.fit(X_train_scaled, y_train)
    
    # 예측 및 평가
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }
    
    print(f"📊 평가 결과:")
    print(f"   - RMSE: {metrics['rmse']:.4f}")
    print(f"   - MAE: {metrics['mae']:.4f}")
    print(f"   - R²: {metrics['r2']:.4f}")
    
    # 피처 중요도 출력
    if hasattr(model, "feature_importances_"):
        print(f"\n📈 피처 중요도:")
        for feat, imp in zip(X.columns, model.feature_importances_):
            print(f"   - {feat}: {imp:.4f}")
    
    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "feature_names": list(X.columns)
    }


def save_model(result: dict, model_name: str = "safety_model"):
    """모델과 스케일러 저장"""
    model_path = MODEL_DIR / f"{model_name}.joblib"
    scaler_path = MODEL_DIR / f"{model_name}_scaler.joblib"
    
    joblib.dump(result["model"], model_path)
    joblib.dump(result["scaler"], scaler_path)
    
    # 메타데이터 저장
    metadata = {
        "feature_names": result["feature_names"],
        "metrics": result["metrics"]
    }
    metadata_path = MODEL_DIR / f"{model_name}_metadata.joblib"
    joblib.dump(metadata, metadata_path)
    
    print(f"\n✅ 모델 저장 완료:")
    print(f"   - 모델: {model_path}")
    print(f"   - 스케일러: {scaler_path}")
    print(f"   - 메타데이터: {metadata_path}")


def main():
    """메인 학습 파이프라인"""
    print("=" * 50)
    print("🚀 모델 학습 시작")
    print("=" * 50)
    
    # 학습 데이터 경로 (전처리 후 생성된 피처 데이터)
    training_data_path = DATA_DIR / "processed" / "training_features.csv"
    
    if not training_data_path.exists():
        print(f"❌ 학습 데이터를 찾을 수 없습니다: {training_data_path}")
        print("💡 먼저 preprocessing.py를 실행하여 피처 데이터를 생성해주세요.")
        
        # 데모용 샘플 데이터 생성
        print("\n📝 데모용 샘플 데이터를 생성합니다...")
        create_sample_training_data()
        
    # 데이터 로드
    df = load_training_data(training_data_path)
    
    # 피처 준비
    X, y = prepare_features(df)
    
    # 모델 학습
    result = train_model(X, y, model_type="random_forest")
    
    # 모델 저장
    save_model(result, "safety_model")
    
    print("\n" + "=" * 50)
    print("✅ 학습 완료!")
    print("=" * 50)


def create_sample_training_data():
    """데모용 샘플 학습 데이터 생성"""
    np.random.seed(42)
    n_samples = 1000
    
    # 샘플 피처 생성
    data = {
        "latitude": np.random.uniform(37.4, 37.6, n_samples),
        "longitude": np.random.uniform(126.8, 127.1, n_samples),
        "dist_to_streetlight": np.random.exponential(30, n_samples),
        "dist_to_police": np.random.exponential(500, n_samples),
        "dist_to_main_road": np.random.exponential(100, n_samples),
        "streetlight_count_100m": np.random.poisson(3, n_samples)
    }
    
    # 안전 점수 계산 (간단한 공식)
    data["safety_score"] = (
        100 
        - np.minimum(20, data["dist_to_streetlight"] / 5)
        - np.minimum(30, data["dist_to_police"] / 50)
        - np.minimum(20, data["dist_to_main_road"] / 10)
        + np.minimum(15, data["streetlight_count_100m"] * 3)
        + np.random.normal(0, 5, n_samples)  # 노이즈 추가
    )
    data["safety_score"] = np.clip(data["safety_score"], 0, 100)
    
    df = pd.DataFrame(data)
    
    # 저장
    output_dir = DATA_DIR / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_features.csv"
    df.to_csv(output_path, index=False)
    
    print(f"✅ 샘플 데이터 생성 완료: {output_path}")


if __name__ == "__main__":
    main()
