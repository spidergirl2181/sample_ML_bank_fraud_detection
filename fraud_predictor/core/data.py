import os
import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_postgres():
    import psycopg2
    import os

    config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
        "dbname": os.getenv("DB_NAME", "fraud_db"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
    }
    table = os.getenv("DB_TABLE", "fraud_time_series")

    query = f"""
        SELECT time_bucket, avg_age, is_1st_ratio, avg_amount, txn_velocity, fraud_rate
        FROM {table}
        ORDER BY time_bucket
    """

    with psycopg2.connect(**config) as conn:
        return pd.read_sql(query, conn)