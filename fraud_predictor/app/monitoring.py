from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "fraud_predict_requests_total", "Total prediction requests"
)

REQUEST_LATENCY = Histogram(
    "fraud_predict_latency_seconds", "Prediction latency"
)