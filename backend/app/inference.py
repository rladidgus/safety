"""
추론 모듈
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "backend" / "models"

class SafetyPredictor:
    def __init__(self, model_name="safety_model"):
        pass
    def predict(self, features):
        return 50.0

if __name__ == "__main__":
    pass
