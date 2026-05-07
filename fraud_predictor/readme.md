===== 
fraud_predictor/
│
├── app/
│   ├── main.py          # FastAPI entrypoint
│   ├── schemas.py       # request/response models
│   ├── service.py       # business logic (predict)
│   ├── model_loader.py  # load model + cache
│   ├── monitoring.py    # metrics (Prometheus)
│
├── core/
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   ├── forecast.py
│   ├── config.py
│
├── artifacts/
│   └── model.pkl
│
├── train.py
├── inference.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml

=====
Run system with Docker daemon & API call:

# train model 
python train.py

# run API
docker-compose up --build

# test API
curl http://localhost:8000/health


# Prometheus scrap
http://localhost:8000/metrics

# Predict
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "data": [
    {
      "average_age": 35,
      "is_1st_ratio": 0.2,
      "avg_amount": 100,
      "txn_velocity": 5
    }
  ]
}'

=======
Run system in local terminal:
python3 inference.py --csv test/fraud_time_series_fdp.csv --horizon 24