import joblib
from functools import lru_cache

MODEL_PATH = "artifacts/model.pkl"

@lru_cache()
def get_model():
    return joblib.load(MODEL_PATH)