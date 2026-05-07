import numpy as np
import pandas as pd
from datetime import timedelta
from core.config import FEATURES


def build_future_features(df: pd.DataFrame, days: int):
    df = df.copy()
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])

    last_ts = df["time_bucket"].max()
    df["hour"] = df["time_bucket"].dt.hour

    hourly = df.groupby("hour")[FEATURES].median()

    rows = []
    for i in range(1, days * 24 + 1):
        ts = last_ts + timedelta(hours=i)
        h = ts.hour

        row = {"time_bucket": ts}

        for f in FEATURES:
            if h in hourly.index:
                val = hourly.loc[h, f]
            else:
                val = df[f].median()

            # constraints
            if f == "is_1st_ratio":
                val = np.clip(val, 0, 1)

            row[f] = val

        rows.append(row)

    return pd.DataFrame(rows)