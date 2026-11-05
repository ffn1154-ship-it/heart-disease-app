"""Self-check: exercises the API endpoints in-process. Run: python verify.py"""
from fastapi.testclient import TestClient

from app.main import app

SAMPLE = {
    "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212, "fbs": 0,
    "restecg": 1, "thalach": 168, "exang": 0, "oldpeak": 1.0, "slope": 2,
    "ca": 2, "thal": 3,
}

with TestClient(app) as client:
    assert client.get("/health").json() == {"status": "healthy"}

    info = client.get("/info").json()
    assert info["model_type"] == "RandomForestClassifier"
    assert len(info["features"]) == 13

    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction_code"] in (0, 1)
    assert body["heart_disease"] == bool(body["prediction_code"] == 1)

    assert client.post("/predict", json={**SAMPLE, "age": "not a number"}).status_code == 422

print("OK", body)
