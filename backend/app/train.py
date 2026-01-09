"""
모델 학습 모듈
"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
if not DATA_DIR.exists():
    DATA_DIR = PROJECT_ROOT / "backend" / "data"
MODEL_DIR = PROJECT_ROOT / "backend" / "models"
MODEL_DIR.mkdir(exist_ok=True)

def train_and_save_model():
    print("Training model...")
    pass

if __name__ == "__main__":
    train_and_save_model()
