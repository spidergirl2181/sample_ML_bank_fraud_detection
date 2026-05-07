import numpy as np
from typing import Dict, Any

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error

from lightgbm import LGBMRegressor

from core.config import RANDOM_STATE


# ─────────────────────────────────────────────
# Model config (production-ready default)
# ─────────────────────────────────────────────
def get_model() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=2000,          # large + early stopping
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def evaluate_metrics(y_true, y_pred) -> Dict[str, float]:
    eps = 1e-6
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

    return {
        "r2": float(r2),
        "mae": float(mae),
        "mape": float(mape),
    }


# ─────────────────────────────────────────────
# Train + validation (time-series aware)
# ─────────────────────────────────────────────
def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
) -> Dict[str, Any]:

    model = get_model()

    from lightgbm import early_stopping, log_evaluation

    model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="l2",
    callbacks=[
        early_stopping(100),
        log_evaluation(50)
    ]
)

    y_pred = model.predict(X_val)
    metrics = evaluate_metrics(y_val, y_pred)

    return {
        "model": model,
        "metrics": metrics,
        "best_iteration": model.best_iteration_,
    }


# ─────────────────────────────────────────────
# TimeSeries Cross Validation
# ─────────────────────────────────────────────
def time_series_cv(X, y, n_splits=5):

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        res = train_model(X_tr, y_tr, X_val, y_val)

        scores.append(res["metrics"]["r2"])

    return {
        "fold_r2": scores,
        "mean_r2": float(np.mean(scores)),
        "std_r2": float(np.std(scores)),
    }


# ─────────────────────────────────────────────
# Final training (full data)
# ─────────────────────────────────────────────
def train_full(X, y) -> LGBMRegressor:
    model = get_model()

    from lightgbm import log_evaluation

    model.fit(
        X,
        y,
        callbacks=[log_evaluation(0)]
        # verbose=False,
    )

    return model