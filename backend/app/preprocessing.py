"""
데이터 전처리 모듈
"""
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
if not DATA_DIR.exists():
    DATA_DIR = PROJECT_ROOT / "backend" / "data"

PROCESSED_DIR = DATA_DIR / "processed"

def preprocess_all_data():
    """전체 데이터 전처리 파이프라인 (Placeholder)"""
    print("Pre-processing data...")
    pass

if __name__ == "__main__":
    preprocess_all_data()
