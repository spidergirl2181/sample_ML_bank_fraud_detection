from fastapi import FastAPI
from fastapi.responses import JSONResponse
import time

from app.schemas import PredictRequest, PredictResponse
from app.service import predict
from app.monitoring import REQUEST_COUNT, REQUEST_LATENCY

from prometheus_client import generate_latest
from fastapi.responses import Response

app = FastAPI(title="Fraud Rate Predictor API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_api(req: PredictRequest):
    REQUEST_COUNT.inc()

    start = time.time()
    preds = predict(req.data)
    latency = time.time() - start

    REQUEST_LATENCY.observe(latency)

    return PredictResponse(predictions=preds)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")