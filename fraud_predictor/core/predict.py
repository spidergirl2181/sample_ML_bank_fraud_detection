import joblib
import pandas as pd

from core.forecast import build_future_features
from core.config import FEATURES


def main(days=7):
    model = joblib.load("artifacts/model.pkl")

    df = pd.read_csv("data.csv")
    future = build_future_features(df, days)

    X = future[FEATURES]
    future["predicted_fraud_rate"] = model.predict(X)

    print(future.head())


if __name__ == "__main__":
    main()