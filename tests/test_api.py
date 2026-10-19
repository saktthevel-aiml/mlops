from fastapi.testclient import TestClient

from src import api


def test_health_without_model(monkeypatch):
    monkeypatch.setattr(api, "_model", None)
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "model_not_loaded"


def test_schema_has_13_features():
    client = TestClient(api.app)
    response = client.get("/schema")
    assert response.status_code == 200
    assert len(response.json()["features"]) == 13


def test_predict_returns_503_when_model_missing(monkeypatch):
    monkeypatch.setattr(api, "_model", None)
    client = TestClient(api.app)
    payload = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3,
        "slope": 1, "ca": 0, "thal": 3
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 503
