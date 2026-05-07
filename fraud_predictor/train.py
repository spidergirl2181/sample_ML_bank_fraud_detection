import joblib

from core.data import load_csv
from core.features import validate, add_time_features, add_lag_features
from core.model import train_model, train_full, time_series_cv
from core.config import FEATURES, TARGET, TEST_SIZE


def main(path: str):
    df = load_csv(path)

    df = validate(df)
    df = add_time_features(df)
    df = add_lag_features(df)

    split = int(len(df) * (1 - TEST_SIZE))

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # 1. Train + validation
    res = train_model(X_train, y_train, X_val, y_val)
    print("Validation metrics:", res["metrics"])

    # 2. CV (robustness)
    cv = time_series_cv(X_train, y_train)
    print("CV:", cv)

    # 3. Train full model
    model = train_full(X, y)

    joblib.dump(model, "artifacts/model.pkl")

    print("Model saved → artifacts/model.pkl")


if __name__ == "__main__":
    main("test/fact_hourly_fraud_metrics_202605071013.csv")