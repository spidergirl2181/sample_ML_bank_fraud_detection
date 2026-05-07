#!/usr/bin/env python3
"""
Local Forecast (NO API call)

Usage:
  python predict_local.py --horizon 24
  python predict_local.py --csv data.csv --horizon 48
"""

import argparse
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import timedelta

MODEL_PATH = "artifacts/model.pkl"

FEATURES = ["avg_age", "is_1st_ratio", "avg_amount", "txn_velocity"]


# ─────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"❌ Load model failed: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path)

    if "time_bucket" in df.columns:
        df["time_bucket"] = pd.to_datetime(df["time_bucket"])
        df = df.sort_values("time_bucket")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# Build future features 
# ─────────────────────────────────────────────
def build_future(df, horizon):
    df = df.copy()

    if "time_bucket" in df.columns:
        df["hour"] = df["time_bucket"].dt.hour
        last_ts = df["time_bucket"].max()
    else:
        df["hour"] = df.index % 24
        last_ts = pd.Timestamp.now()

    lookback = min(len(df), 24 * 7)
    recent = df.tail(lookback)

    hourly_median = recent.groupby("hour")[FEATURES].median()
    hourly_std = recent.groupby("hour")[FEATURES].std().fillna(0)

    rng = np.random.RandomState(42)

    rows = []
    for i in range(1, horizon + 1):
        ts = last_ts + timedelta(hours=i)
        h = ts.hour

        row = {"time_bucket": ts}

        for f in FEATURES:
            base = hourly_median.loc[h, f]
            std = hourly_std.loc[h, f]
            noise = rng.normal(0, std * 0.3) if std > 0 else 0
            row[f] = max(0, base + noise)

        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────
def predict(model, df):
    preds = model.predict(df[FEATURES])
    return np.clip(preds, 0, None)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="historical data")
    parser.add_argument("--horizon", type=int, default=24,
                        help="number of future time buckets (default=24)")
    args = parser.parse_args()

    df = load_data(args.csv)
    model = load_model()

    # Generate future
    future_df = build_future(df, args.horizon)

    # Predict
    future_df["fraud_rate_pred"] = predict(model, future_df)

    # Output
    print("\n📊 Forecast:")
    print(future_df[["time_bucket", "fraud_rate_pred"]].head(20))

    # Summary
    print("\n📈 Summary:")
    print(f"  avg: {future_df['fraud_rate_pred'].mean():.6f}")
    print(f"  max: {future_df['fraud_rate_pred'].max():.6f}")
    print(f"  min: {future_df['fraud_rate_pred'].min():.6f}")

    # Save optional
    future_df.to_csv("forecast_output.csv", index=False)
    print("\n💾 Saved → forecast_output.csv")


if __name__ == "__main__":
    main()
