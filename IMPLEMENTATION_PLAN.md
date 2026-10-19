# Assignment 01 implementation and completion plan

This plan maps the project to the AIMLCZG523 MLOps Assignment 01 brief. The pipeline has now been run locally, the Docker API is running, and a two-replica Minikube deployment has been verified through port-forwarding. The remaining work is primarily GitHub Actions, any required LoadBalancer proof, screenshots, the final report, and the video. Treat evidence items as pending until you capture and save them yourself.

## Assignment crosswalk

| Assignment requirement | Current project coverage | Remaining completion evidence |
|---|---|---|
| Data acquisition and EDA (5 marks) | `src/data.py` downloads and cleans the UCI Cleveland data; the notebook ran locally and generated class-balance, feature-distribution, and correlation plots. The cleaned CSV is allowed to be included in the repository. | Record the actual row count, missing-value handling, and your observations from all three plots in the report; capture the plot evidence. |
| Feature engineering and model development (8 marks) | The local run compared Logistic Regression and Random Forest using stratified 5-fold grid search. Logistic Regression was selected on mean CV ROC-AUC (0.9001); its held-out accuracy/precision/recall/ROC-AUC were 0.8525/0.8065/0.8929/0.9578. | Use these locally generated results and your own interpretation in the report. Rerun if you change the data or tuning. |
| Experiment tracking (5 marks) | MLflow logs both model runs, parameters, cross-validation and held-out metrics, model artifacts, reports, EDA plots, confusion matrices, and ROC curves. | Open the local MLflow UI and capture the actual experiment runs. |
| Packaging and reproducibility (7 marks) | `requirements.txt`, the full preprocessing/model pipeline, saved joblib model, download script, and notebook are present; the notebook and model run completed locally. | Verify from a fresh virtual environment and record the exact Python/dependency versions used. |
| CI/CD and automated testing (8 marks) | Seven Pytest tests and Ruff pass locally; GitHub Actions is configured for lint, tests, training, Docker build, and uploaded workflow artifacts. | This folder is not currently a Git repository. Create/push your own GitHub repository, then capture a successful Actions run and inspect its uploaded artifacts. |
| Containerization (5 marks) | The Docker image was built and is running locally. `/health`, `/predict`, and `/metrics` were verified successfully through the container. | Capture the actual running container and successful endpoint responses. |
| Production deployment (7 marks) | The local Minikube cluster has two running API replicas; health, prediction, and metrics were verified using a port-forward. The manifest defines a LoadBalancer Service. | For a LoadBalancer IP on Minikube, run `minikube tunnel` and verify the assigned address; otherwise document the verified port-forward instructions. For cloud, replace the local image with a pushed registry image. |
| Monitoring and logging (3 marks) | The API's Prometheus request counter and structured prediction logs were verified from the running Kubernetes deployment. | Capture the live metrics and logs. Add Prometheus/Grafana only if you choose that route; the assignment also accepts a simple API metrics/logs dashboard. |
| Documentation and reporting (2 marks) | Architecture, deployment documentation, and a report template are present. | Complete a personalized 10-page report (PDF/DOCX), add actual screenshots in `screenshots/`, include your repository link and local access instructions, and record the short end-to-end video. |

## Ordered work plan

### 1. Prepare the local environment — complete

- The existing Python 3.12 environment passed tests and Ruff.
- Colima is running and Docker responds.

### 2. Regenerate the data, model, and evidence — complete locally

- The notebook and training pipeline ran; `artifacts/metrics.json` records the CV-based selection and actual held-out metrics.
- Review the generated plots and MLflow UI before writing the report.

### 3. Verify application quality and local serving — complete

- Seven pytest tests and Ruff pass.
- FastAPI and the Docker container both returned successful health and sample-prediction responses; `/metrics` reported the request counter.

### 4. Complete deployment proof — running locally

- Minikube is active and is the current Kubernetes context; both API replicas are running.
- Port-forwarded health, prediction, metrics, and logs have been verified. Start `minikube tunnel` only if you need to demonstrate an assigned LoadBalancer IP.
- Capture the actual deployment evidence. No public cloud deployment has been performed.

### 5. Publish, report, and record — pending

- Initialize a Git repository in this project folder, create your own GitHub repository, push the project, and verify the Actions workflow and artifacts.
- Fill in [report_template.md](report_template.md) using your own explanations and freshly captured results. Export it as a polished 10-page PDF or DOCX.
- Save genuine screenshots in `screenshots/` and record the end-to-end walkthrough to `reports/demo-video.mov` (or another clearly named video file).
- Review the submission checklist below before submitting.

## Completion checklist

- [x] Cleaned 14-column dataset is present and the notebook generated EDA plots.
- [x] Local run created model, metrics, and MLflow runs.
- [x] Metrics report two models, selected hyperparameters, cross-validation ROC-AUC, and held-out accuracy/precision/recall/ROC-AUC.
- [x] Pytest (7 tests) and Ruff pass locally.
- [x] FastAPI sample request succeeds locally.
- [x] Docker image builds and the running container passes health and prediction checks.
- [x] Minikube deployment is running and endpoints were verified via port-forward.
- [x] API request metrics and logs were verified.
- [ ] Capture screenshots for plots, MLflow, Actions, Docker/API, Kubernetes, and monitoring.
- [ ] If required, run `minikube tunnel` and capture a LoadBalancer IP.
- [ ] GitHub Actions has a successful run with its uploaded artifacts.
- [ ] The personalized 10-page report, real screenshots, repository link, local access instructions, and short video are ready.

## Integrity and safety

This is a small educational dataset and this API is not a medical device. Do not use it for diagnosis, treatment, or real patient decisions. Explain your own observations and tuning choices; do not present seeded artifacts, old notebook outputs, or unverified cloud resources as results from a fresh run.
