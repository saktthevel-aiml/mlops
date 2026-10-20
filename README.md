# MLOps Assignment 1 — Heart Disease Classification

Course: AIMLCZG523 — Machine Learning Operations.
End-to-end workflow: data download, EDA, model training with MLflow tracking, tests and CI, a FastAPI prediction service, Docker, Kubernetes (Minikube) and Prometheus/Grafana monitoring.

> Educational project only, not a medical device.

## Repository layout

```text
src/data.py            dataset download + cleaning
src/modeling.py        preprocessing, CV, model comparison, MLflow logging
src/predict.py         command-line inference
src/api.py             FastAPI: /health /schema /predict /metrics
notebooks/             EDA + modeling notebook
tests/                 pytest unit + API tests
k8s/deployment.yaml    Deployment (2 replicas) + LoadBalancer Service
monitoring/            Prometheus + Grafana config
Dockerfile, docker-compose.yml, docker-compose.monitoring.yml
.github/workflows/ci.yml
reports/, screenshots/ report and evidence
```

## Prerequisites

- Python 3.11 or 3.12, git
- Docker (Docker Desktop, or Colima on macOS: `colima start`)
- Minikube + kubectl (macOS: `brew install minikube kubectl`)

Run every command from the repository root.

## Step 1 — Virtual Environment and install requirements

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

## Step 2 — Download data

```bash
python -m src.data --output data/heart_disease.csv
```

## Step 3 — Train and track with MLflow

```bash
python -m src.modeling --data data/heart_disease.csv --model-dir artifacts --tracking-uri ./mlruns
```

Result: Logistic Regression and Random Forest are compared by mean 5-fold CV ROC-AUC; the best is saved to `artifacts/model.joblib`, with metrics in `artifacts/`. Typical result: Logistic Regression selected (CV AUC about 0.90, test accuracy about 0.85).

> Training clears stale MLflow folders named `0`, `1` and `.trash` inside `./mlruns`. Check `git status` after training.

View the runs:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 and screenshot the experiment and run comparison.

## Step 4 — Lint and tests

```bash
ruff check .
python -m pytest -q
```

Result: `All checks passed!` and `12 passed`.

## Step 5 — Run the API locally

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":1,"ca":0,"thal":3}'
curl http://127.0.0.1:8000/metrics
```

Result: `{"status":"ok"}`, then a JSON prediction with class and confidence, then Prometheus counters. Swagger UI is at http://127.0.0.1:8000/docs. Stop the server with Ctrl+C.

Command-line inference without the API:

```bash
python -m src.predict --model artifacts/model.joblib --input data/sample_input.json
```

## Step 6 — Docker

```bash
docker build -t heart-disease-api:local .
docker run --rm -p 8000:8000 heart-disease-api:local
```

Or `docker compose up --build -d` (then `docker compose down`). Repeat the curl checks from Step 5. The image expects `artifacts/model.joblib` to exist, so complete Step 3 first.

## Step 7 — Kubernetes (Minikube)

```bash
minikube start --driver=docker --cpus=2 --memory=4096
minikube kubectl config use-context minikube
minikube image load heart-disease-api:local
minikube kubectl -- apply -f k8s/deployment.yaml
minikube kubectl get pods,svc
minikube kubectl port-forward svc/heart-disease-api 8081:80
```

Expected: 2 pods `Running`. In another terminal, `curl http://127.0.0.1:8081/health` and repeat the `/predict` call on port 8081. `kubectl logs deploy/heart-disease-api` shows request logs. For a LoadBalancer IP run `minikube tunnel` (needs sudo). Clean up with `kubectl delete -f k8s/deployment.yaml` and `minikube stop`.

## Step 8 — Monitoring (Prometheus + Grafana)

```bash
export GRAFANA_ADMIN_PASSWORD='choose-a-password'
docker compose -f docker-compose.monitoring.yml up --build -d
```

- API: http://localhost:8000
- Prometheus: http://localhost:9090 (Status → Targets should show `api` UP)
- Grafana: http://localhost:3000 (user `admin`, the password you set, default `change-me-locally`). The Prometheus datasource and the **Heart Disease API** dashboard are provisioned automatically: open **Dashboards → Heart Disease API** (or http://localhost:3000/d/heart-disease-api). No manual creation or import is needed. If it doesn't appear, run `docker compose -f docker-compose.monitoring.yml up -d --force-recreate grafana`.

Send a few `/predict` requests, then query `prediction_requests_total` in Prometheus or Grafana Explore. Stop with `docker compose -f docker-compose.monitoring.yml down`.

## Step 9 — CI/CD

`.github/workflows/ci.yml` runs Ruff, pytest, dataset download, training and uploads artifacts on push and pull requests. Push to `main`, open the repository's **Actions** tab, and screenshot a green run.

## Step 10 — Report and demo video

1. Notebook: `jupyter nbconvert --to notebook --execute notebooks/heart_disease_mlops.ipynb` (or run it in VS Code) for EDA plots.
2. Put screenshots (MLflow, Actions, API, Docker, pods, Grafana) in `screenshots/`.
3. Write the ~10-page report in `reports/` using `report_template.md`, in your own words.
4. Record a short demo (macOS: Shift-Command-5) following [LOCAL_DEMO_GUIDE.md](LOCAL_DEMO_GUIDE.md): training, MLflow UI, tests, API call, Docker, Kubernetes, monitoring, CI run.

See also [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ModuleNotFoundError: src` | Run from the repo root with `python -m pytest` / `python -m src.…` |
| MLflow `FallbackAsyncAdaptedQueuePool` error | `pip install "sqlalchemy>=2,<2.1"` (already pinned) |
| `log_model() got unexpected keyword 'name'` | Needs `mlflow>=3,<4` (already pinned) |
| API returns 503 / model missing | Run Step 3 first |
| Pods `ErrImagePull` | Run `minikube image load heart-disease-api:local` |
| `docker build` cannot reach daemon | Start Docker (`colima start`) |
| `port is already allocated` (8000) | Another container uses it: `docker ps`, then `docker rm -f <name>` (or `docker compose down` in the folder that started it), or map another port, e.g. `-p 8010:8000` |
| `WARN Docker Compose requires buildx plugin` | Harmless (legacy builder is used). To remove: `brew install docker-buildx && mkdir -p ~/.docker/cli-plugins && ln -sfn "$(brew --prefix)/opt/docker-buildx/bin/docker-buildx" ~/.docker/cli-plugins/docker-buildx` |
