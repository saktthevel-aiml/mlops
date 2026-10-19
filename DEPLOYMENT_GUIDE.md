# CI/CD, Docker, Kubernetes and monitoring guide

## A. Prepare and run locally

1. Install Python 3.11 and Docker.
2. Create/activate a virtual environment and install `requirements-dev.txt`.
3. Run:
   ```bash
   python -m src.data --output data/heart_disease.csv
   python -m src.modeling --data data/heart_disease.csv --model-dir artifacts --tracking-uri ./mlruns
   python -m pytest -q
   ruff check .
   ```
4. Inspect `artifacts/metrics.json`, `artifacts/plots/`, and the MLflow UI. Use the metrics that actually appear; do not copy expected numbers from elsewhere.
5. Run the API with `uvicorn src.api:app --reload --port 8000`, then open `http://127.0.0.1:8000/docs`.

## B. GitHub Actions CI

1. Create an empty GitHub repository.
2. Add this project, commit, and push to `main`:
   ```bash
   git init
   git add .
   git commit -m "Build reproducible heart disease MLOps pipeline"
   git branch -M main
   git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
   git push -u origin main
   ```
3. Open **Actions** in GitHub. The workflow runs on push, pull request, or manual dispatch.
4. The job fails if lint/tests/training/Docker build fail. Download the run artifact to inspect test results, trained model, MLflow files, and dataset.
5. For a meaningful test of CI, introduce a temporary failing test on a branch, push it, verify the run fails, then revert the test. Capture screenshots of a successful run and a failed run.

## C. Docker proof

A model artifact must exist before building because the Dockerfile copies `artifacts/`.

```bash
docker build -t heart-disease-api:local .
docker run --rm -p 8000:8000 heart-disease-api:local
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

Test a prediction:
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":1,"ca":0,"thal":3}'
```

If the response says the model is not loaded, retrain and rebuild the image. Keep the JSON feature names and ranges aligned with `/schema` and the selected dataset.

## D. Push to a container registry (example: Docker Hub)

```bash
docker login
docker tag heart-disease-api:local YOUR_DOCKERHUB_USER/heart-disease-api:v1
docker push YOUR_DOCKERHUB_USER/heart-disease-api:v1
```

Replace the image in `k8s/deployment.yaml` with the exact registry path and tag. Do not commit registry credentials.

## E. Deploy to Kubernetes

Prerequisites: a running cluster and `kubectl` configured for that cluster. A cloud LoadBalancer may incur costs.

```bash
kubectl apply -f k8s/deployment.yaml
kubectl rollout status deployment/heart-disease-api
kubectl get pods,svc
kubectl get svc heart-disease-api
```

Wait for the external IP/hostname:
```bash
kubectl get svc heart-disease-api -w
```
Then:
```bash
curl http://EXTERNAL_ADDRESS/health
curl http://EXTERNAL_ADDRESS/docs
curl http://EXTERNAL_ADDRESS/metrics
```

If deploying to Minikube, use `minikube tunnel` for LoadBalancer service support or change the Service type to NodePort. For a public service, consider authentication, TLS, rate limits, network restrictions, and secrets management before exposing it.

## F. Monitoring and logs

The API emits JSON log events to stdout and exports Prometheus counters and latency histograms at `/metrics`. For a basic proof:
```bash
kubectl logs deployment/heart-disease-api --tail=100
curl http://EXTERNAL_ADDRESS/metrics
```

For a full Prometheus/Grafana demonstration, configure Prometheus to scrape the API's `/metrics` endpoint. In Kubernetes, add the scrape annotations or a `ServiceMonitor` if Prometheus Operator is installed. Build a Grafana panel for `prediction_requests_total` and request latency. Screenshot the dashboard and describe the metric names and scrape configuration in the report.

## G. Evidence screenshots checklist

Save real screenshots in `screenshots/`:
- `01_eda_class_balance.png`
- `02_eda_correlation.png`
- `03_mlflow_runs.png`
- `04_github_actions_success.png`
- `05_github_actions_failure.png`
- `06_docker_container.png`
- `07_api_docs_predict.png`
- `08_kubernetes_pods_service.png`
- `09_deployed_health.png`
- `10_monitoring_dashboard.png`

Do not create mock screenshots; capture them from your own run.
