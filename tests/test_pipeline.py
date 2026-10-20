import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import api
from src.data import FEATURES, TARGET
from src.modeling import model_candidates, train
from src.predict import predict_record


def synthetic_frame(rows: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(rows, len(FEATURES))), columns=FEATURES)
    frame[TARGET] = (frame["age"] + rng.normal(scale=0.5, size=rows) > 0).astype(int)
    frame.loc[0, "chol"] = np.nan
    return frame


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    previous = os.getcwd()
    os.chdir(out)  # MLflow's default artifact root is relative to the working directory
    try:
        summary = train(synthetic_frame(), out / "artifacts", tracking_uri=str(out / "mlruns"))
    finally:
        os.chdir(previous)
    return out / "artifacts", summary


def test_training_writes_model_and_metrics(trained):
    artifacts, summary = trained
    assert (artifacts / "model.joblib").exists()
    assert (artifacts / "metrics.json").exists()
    assert summary["selected_model"] in model_candidates()
    assert len(summary["results"]) == 2
    for result in summary["results"]:
        for key in ("test_accuracy", "test_precision", "test_recall", "test_roc_auc", "cv_best_roc_auc"):
            assert 0.0 <= result[key] <= 1.0


def test_selection_uses_cross_validation_auc(trained):
    _, summary = trained
    best = max(summary["results"], key=lambda r: r["cv_best_roc_auc"])
    assert summary["selected_model"] == best["model"]


def test_predict_record_and_api_success(trained, monkeypatch):
    import joblib

    artifacts, _ = trained
    model = joblib.load(artifacts / "model.joblib")
    record = {f: 0.5 for f in FEATURES}
    result = predict_record(model, record)
    assert result["prediction"] in (0, 1)
    assert 0.5 <= result["confidence"] <= 1.0

    monkeypatch.setattr(api, "_model", model)
    response = TestClient(api.app).post("/predict", json={**record, "age": 50, "sex": 1, "cp": 3,
                                                          "trestbps": 120, "chol": 200, "fbs": 0,
                                                          "restecg": 0, "thalach": 150, "exang": 0,
                                                          "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 3})
    assert response.status_code == 200
    assert {"prediction", "confidence"} <= set(response.json())


def test_predict_record_rejects_missing_features():
    with pytest.raises(ValueError, match="Missing features"):
        predict_record(object(), {"age": 1})


def test_api_rejects_out_of_range_input():
    response = TestClient(api.app).post("/predict", json={f: 0 for f in FEATURES} | {"age": 500})
    assert response.status_code == 422
