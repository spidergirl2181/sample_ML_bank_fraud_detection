import pandas as pd
from core.config import FEATURES, TARGET


def validate(df: pd.DataFrame) -> pd.DataFrame:
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=required)

    # clip instead of drop (avoid losing spikes)
    q1 = df[TARGET].quantile(0.01)
    q3 = df[TARGET].quantile(0.99)
    df[TARGET] = df[TARGET].clip(q1, q3)

    return df.reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "time_bucket" in df.columns:
        df["time_bucket"] = pd.to_datetime(df["time_bucket"])
        df["hour"] = df["time_bucket"].dt.hour
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df["fraud_lag_1"] = df["fraud_rate"].shift(1)
    df["fraud_lag_24"] = df["fraud_rate"].shift(24)
    return df.dropna()