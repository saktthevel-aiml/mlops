# Local setup and assignment demo guide

Use this guide from the project root. Run commands yourself and capture only real outputs. The sample values below are synthetic and are provided only to exercise the educational API; they are not clinical advice.

## 1. Prepare Python and regenerate the pipeline

The existing project virtual environment is Python 3.12; the GitHub Actions workflow uses Python 3.11. The dependency ranges support the local environment, but note which interpreter you use in the report. To retain the current environment, do not recreate `.venv` over it.

```bash
cd "/Users/skv695@apac.comcast.com/Documents/vs_code/mlops/mlops_heart_disease_assignment"
source .venv/bin/activate
python --version
python -m pip install -r requirements-dev.txt
python -m src.data --output data/heart_disease.csv
python -m src.modeling --data data/heart_disease.csv --model-dir artifacts --tracking-uri ./mlruns
```

Training uses a stratified 80/20 split and 5-fold cross-validation. It selects between the tuned Logistic Regression and Random Forest candidates using mean cross-validation ROC-AUC, then reports held-out test metrics. Review the actual values in `artifacts/metrics.json`; do not copy any example values into the report.

## 2. Run quality checks

```bash
python -m pytest -q
ruff check .
```

Both commands should finish successfully before recording the demo.

## 3. Start MLflow

In a terminal from the project root, with the virtual environment active:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`, select the `uci-heart-disease-classification` experiment, and confirm both model runs and their parameters/metrics. Keep this process running while you capture the MLflow segment.

## 4. Start the API and test it

In another terminal from the project root:

```bash
source .venv/bin/activate
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` to show the interactive API documentation. From a third terminal, check health, schema, a prediction, and metrics:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/schema
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":1,"ca":0,"thal":3}'
curl http://127.0.0.1:8000/metrics
```

Show that the prediction response contains a class, confidence, and the educational-use disclaimer. Confidence is not a calibrated personal risk estimate.

## 5. Build and run the Docker container

The Docker CLI is installed in the current environment, but the Docker engine must be running. Start Docker Desktop or the configured Colima runtime, then verify it before continuing:

```bash
docker info
docker compose up --build
```

With Compose running, use a separate terminal to call the same health, prediction, and metrics URLs at `http://127.0.0.1:8000`. Stop Compose with `Ctrl+C` when finished. If port 8000 is already in use by the local API, stop that process first or use a different host port in `docker-compose.yml`.

## 6. Optional local Kubernetes deployment

The Kubernetes manifest defaults to the local image `heart-disease-api:local`. A local Kubernetes environment is not available until you install/start one (for example, Minikube with Docker Desktop or Colima).

Before deploying, inspect the active context:

```bash
kubectl config current-context
```

Use a local context such as `minikube` or Docker Desktop Kubernetes only. Do not apply the manifest while the context points to a remote or otherwise unintended cluster.

For Minikube, after installing and starting its Docker engine:

```bash
minikube start --driver=docker
docker build -t heart-disease-api:local .
minikube image load heart-disease-api:local
```

Confirm `kubectl config current-context` reports Minikube and the manifest uses `heart-disease-api:local` with `imagePullPolicy: IfNotPresent`, then run:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl rollout status deployment/heart-disease-api
kubectl get pods,svc
kubectl port-forward svc/heart-disease-api 8081:80
```

In another terminal, verify `http://127.0.0.1:8081/health`, `/docs`, `/metrics`, and a prediction. `kubectl port-forward` works without exposing a public service. Alternatively, for Minikube's LoadBalancer service, use `minikube tunnel` in a separate terminal and verify its actual address. For a cloud deployment, replace the image with an image you built and pushed to your own registry; review the cloud deployment and cost/security requirements before applying anything.

## 7. Capture assignment evidence

Save genuine, legible screenshots under `screenshots/`. A useful set is:

- EDA class-balance, feature-histogram, and correlation plots.
- MLflow experiment with both candidate runs and recorded metrics.
- Successful local test/lint output and a successful GitHub Actions run.
- FastAPI docs and a successful `/predict` response.
- Running Docker container and successful endpoint checks.
- Local Kubernetes pods/service and a verified local endpoint, if you complete Kubernetes.
- Real `/metrics` output or a dashboard and corresponding request logs.

Complete `report_template.md` in your own words, using actual results and a diagram/screenshots from your run. The assignment asks for a final report of about 10 pages as a PDF or DOCX and a code repository link.

To publish the project, first create an empty GitHub repository in your own account. This project folder is not currently initialized as a Git repository. Then, from the project root:

```bash
git init
git add .
git commit -m "Complete MLOps assignment implementation"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repository>.git
git push -u origin main
```

Replace the remote URL with your repository URL. Confirm that the `.venv`, MLflow runs, and generated model artifacts are excluded, and that `data/heart_disease.csv` is included. After the push, check the **Actions** tab, wait for the workflow to pass, and download its artifacts.

## 8. Short video walkthrough

On macOS, press **Shift-Command-5**, choose a screen-recording option, select the useful screen area, and start recording. Stop with the stop control in the menu bar. Save the original recording to `reports/demo-video.mov` (or another clearly named file). A concise 4–6 minute outline:

| Segment | Show and explain |
|---|---|
| 0:00–0:30 | Assignment objective, project structure, and the educational/non-clinical limitation. |
| 0:30–1:15 | Dataset acquisition/shape and the generated EDA plots; describe two or three observations you verified. |
| 1:15–2:00 | Two model candidates, cross-validation selection, actual test metrics, and the two MLflow runs. |
| 2:00–2:45 | Pytest/Ruff result and, if available, the successful GitHub Actions run and its artifacts. |
| 2:45–4:15 | Running local API or container; demonstrate `/health`, one synthetic `/predict`, and `/metrics`. |
| 4:15–5:15 | If completed, show local Kubernetes rollout/service; otherwise clearly state which deployment step remains. |
| 5:15–5:45 | Briefly recap reproducibility, monitoring, limitations, and the report/repository deliverables. |

Before recording, close unrelated windows and notifications, avoid showing passwords/tokens or private account details, and make sure no real patient data is visible. Do not claim a cloud deployment or monitoring dashboard that you have not actually set up.
