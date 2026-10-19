"""FastAPI inference service with structured request logging and Prometheus metrics."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field

from src.data import FEATURES

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(message)s")
logger = logging.getLogger("heart_disease_api")

REQUESTS = Counter("prediction_requests_total", "Number of prediction requests", ["status"])
LATENCY = Histogram("prediction_request_duration_seconds", "Prediction request latency")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
app = FastAPI(title="Heart Disease Classifier API", version="1.0.0",
              description="Educational demonstration only; not for clinical use.")
_model = None


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: float = Field(ge=1, le=120)
    sex: float = Field(ge=0, le=1, description="Dataset coding: 0/1")
    cp: float = Field(ge=1, le=4, description="Chest-pain category in original dataset")
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(ge=0, le=1000)
    fbs: float = Field(ge=0, le=1)
    restecg: float = Field(ge=0, le=2)
    thalach: float = Field(gt=0, le=300)
    exang: float = Field(ge=0, le=1)
    oldpeak: float = Field(ge=-10, le=20)
    slope: float = Field(ge=1, le=3)
    ca: float = Field(ge=0, le=4)
    thal: float = Field(ge=0, le=7)


@app.on_event("startup")
def load_model() -> None:
    global _model
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        logger.info(json.dumps({"event": "model_loaded", "path": str(MODEL_PATH)}))
    else:
        logger.warning(json.dumps({"event": "model_missing", "path": str(MODEL_PATH)}))


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if _model is not None else "model_not_loaded"}


@app.get("/schema")
def schema() -> dict:
    return {"features": FEATURES, "target": {"0": "absence in dataset", "1": "presence in dataset"}}


@app.post("/predict")
def predict(payload: PredictionInput, request: Request) -> dict:
    if _model is None:
        REQUESTS.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded; train model first.")
    with LATENCY.time():
        values = payload.model_dump()
        row = pd.DataFrame([[values[c] for c in FEATURES]], columns=FEATURES)
        prediction = int(_model.predict(row)[0])
        probabilities = _model.predict_proba(row)[0]
        confidence = float(max(probabilities))
    REQUESTS.labels(status="success").inc()
    logger.info(json.dumps({
        "event": "prediction", "path": request.url.path, "status": "success",
        "prediction": prediction, "confidence": round(confidence, 5),
    }))
    return {
        "prediction": prediction,
        "label": "presence_in_dataset" if prediction == 1 else "absence_in_dataset",
        "confidence": round(confidence, 5),
        "class_probabilities": {str(i): round(float(p), 5) for i, p in enumerate(probabilities)},
        "disclaimer": "Educational output, not a diagnosis or clinical risk estimate.",
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)
