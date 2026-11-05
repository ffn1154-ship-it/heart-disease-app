# Heart Disease Prediction API

FastAPI service that serves a Random Forest classifier trained on the UCI
Cleveland heart disease dataset.

## Structure

```
heart-disease-app/
├── model/heart_model.joblib   # trained model (generated)
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI app
│   └── schemas.py             # request/response models
├── train_model.py             # trains the model
├── verify.py                  # endpoint self-check
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 1. Train the model

```bash
pip install -r requirements.txt
python train_model.py
```

Downloads `heart.csv` on first run and writes `model/heart_model.joblib`.

## 2. Run the API

```bash
uvicorn app.main:app --reload
```

Docs: http://localhost:8000/docs

## 3. Run with Docker

```bash
docker compose up --build
```

The model must exist before building — run step 1 first.

## Endpoints

| Method | Path       | Description                       |
|--------|------------|-----------------------------------|
| GET    | `/health`  | Liveness check                    |
| GET    | `/info`    | Model type and expected features  |
| POST   | `/predict` | Predict heart disease presence    |

### Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,
       "restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,
       "slope":2,"ca":2,"thal":3}'
```

```json
{"heart_disease": false, "prediction_code": 0}
```

`prediction_code` is `1` when heart disease is predicted, `0` otherwise.

## Self-check

```bash
python verify.py
```

Trains nothing; exercises all three endpoints against the saved model.

## Note

This is a demo model. It is not a medical device and must not be used for
diagnosis.
