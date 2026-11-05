from contextlib import asynccontextmanager
import os

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import HeartInput, PredictionResponse

MODEL_PATH = "model/heart_model.joblib"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease using machine learning",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}


@app.get("/info", tags=["Info"])
def model_info():
    return {"model_type": type(model).__name__, "features": FEATURES}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(data: HeartInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    input_data = pd.DataFrame([[
        data.age, data.sex, data.cp, data.trestbps, data.chol,
        data.fbs, data.restecg, data.thalach, data.exang,
        data.oldpeak, data.slope, data.ca, data.thal,
    ]], columns=FEATURES)

    prediction = model.predict(input_data)[0]

    return {
        "heart_disease": bool(prediction == 1),
        "prediction_code": int(prediction),
    }
