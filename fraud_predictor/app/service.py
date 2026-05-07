import numpy as np
from app.model_loader import get_model
from core.config import FEATURES


def preprocess_input(data):
    rows = []
    for d in data:
        row = [getattr(d, f) for f in FEATURES]
        rows.append(row)
    return np.array(rows)


def predict(data):
    model = get_model()
    X = preprocess_input(data)
    preds = model.predict(X)

    # enforce non-negative fraud rate
    return np.clip(preds, 0, None).tolist()