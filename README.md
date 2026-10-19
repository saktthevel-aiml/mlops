# MLOps Assignment 01 — Heart Disease Classification

**Course:** AIMLCZG523 — Machine Learning Operations  
**Goal:** Build a reproducible ML workflow, track experiments, test it automatically, package a prediction API, and deploy/monitor it.

> **Important:** This project is an educational demonstration, not a medical device. Do not use predictions for diagnosis or treatment. Verify every result by running the workflow yourself and personalize the implementation/report to comply with your course's academic-integrity requirements.

## Repository structure

```text
.
├── notebooks/                         # End-to-end EDA and modeling notebook
├── src/
│   ├── data.py                        # UCI dataset download and cleaning
│   ├── modeling.py                    # Preprocessing, CV, model comparison, MLflow
│   └── api.py                         # FastAPI /predict and /metrics
├── tests/                             # Pytest unit and API tests
├── .github/workflows/ci.yml           # Lint, tests, train, Docker build
├── k8s/                               # Kubernetes Deployment and Service
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── report_template.md
└── DEPLOYMENT_GUIDE.md
```

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the assignment crosswalk and remaining evidence, and [LOCAL_DEMO_GUIDE.md](LOCAL_DEMO_GUIDE.md) for local setup, API/container/Kubernetes checks, and the video outline.

## Quick start (Python 3.11 recommended)

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements-dev.txt
python -m src.data --output data/heart_disease.csv
python -m src.modeling --data data/heart_disease.csv --model-dir artifacts --tracking-uri ./mlruns
```

The download script obtains the Cleveland Heart Disease dataset through `ucimlrepo`. Training compares candidates using mean cross-validation ROC-AUC, then reports the selected model's held-out test metrics and saves its full preprocessing/model pipeline to `artifacts/model.joblib`. MLflow runs are stored locally; start the UI with:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000.

## Run the API

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

The API expects the model at `artifacts/model.joblib`. In a separate terminal:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":1,"ca":0,"thal":3}'
```

Useful endpoints:
- `GET /health` — liveness
- `GET /schema` — input feature names and expected types
- `POST /predict` — class and confidence
- `GET /metrics` — lightweight Prometheus-format counters

Confidence is the highest predicted class probability, not a calibrated personal risk estimate.

## Tests, lint, and training

```bash
python -m pytest -q
ruff check .
python -m src.modeling --data data/heart_disease.csv --model-dir artifacts --tracking-uri ./mlruns
```

To build and run with Docker:

```bash
python -m src.data --output data/heart_disease.csv
python -m src.modeling --data data/heart_disease.csv --model-dir artifacts --tracking-uri ./mlruns
docker build -t heart-disease-api:local .
docker run --rm -p 8000:8000 heart-disease-api:local
```

Or use Compose after generating the model:

```bash
docker compose up --build
```

## CI/CD

GitHub Actions runs Ruff, pytest, model training, and a Docker image build on pushes and pull requests. It uploads test/training logs and model artifacts as workflow artifacts. The workflow does **not** deploy publicly by default: configure a registry and cluster credentials only after tests pass and you are ready to deploy.

## Data notes

This starter uses the UCI Heart Disease dataset (Cleveland subset, UCI dataset ID 45) via the `ucimlrepo` Python package. The original target is commonly encoded as 0–4; this project maps `num == 0` to 0 and `num > 0` to 1. Missing feature values are handled by the preprocessing pipeline. The dataset is small and is not representative of all populations.

## Academic integrity and evidence

Use this as a learning scaffold. Run it, inspect the data and plots, perform your own tuning, record the actual MLflow results, add screenshots from your environment, and write the report in your own words. Do not claim cloud deployment or results that you have not performed.
