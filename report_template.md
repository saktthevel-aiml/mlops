# Heart Disease MLOps Assignment Report

## 1. Project objective

This project is an end-to-end MLOps implementation for a heart disease prediction workflow. The objective is to build a reproducible machine-learning pipeline that includes data acquisition, preprocessing, exploratory data analysis (EDA), model development, experiment tracking with MLflow, testing, Docker containerization, and local Kubernetes deployment. The application is intentionally framed as an educational example and is not a medical device.

## 2. Dataset and problem framing

The dataset used is the Cleveland Heart Disease dataset from the UCI Machine Learning Repository. The original problem is a binary classification task: predict whether a patient has heart disease. In this project, the target is transformed to a binary label where 0 means no disease and 1 means disease present.

The project uses the following feature fields:

- age
- sex
- cp
- trestbps
- chol
- fbs
- restecg
- thalach
- exang
- oldpeak
- slope
- ca
- thal

The cleaned dataset file is saved as [data/heart_disease.csv](data/heart_disease.csv). Missing values are handled as part of the input pipeline, and the model is trained on the processed numeric features.

## 3. Data acquisition and EDA

The data pipeline is implemented in [src/data.py](src/data.py). It loads the UCI dataset, validates the feature schema, maps the target to a binary label, and writes a cleaned CSV to disk. The EDA workflow produces plots for the class balance, the feature distributions, and the correlation heatmap. These visualizations are stored under [artifacts/plots](artifacts/plots) and are also mirrored in the screenshot folder.

The generated plots show:

- a reasonably balanced binary target, with the majority class not dominating the dataset;
- the feature distributions are mostly centered and continuous, with a few variables showing skew or outliers;
- the correlation heatmap highlights the strongest relationships between input variables and the target and helps clarify which features are most informative.

## 4. Feature engineering and modeling

The modeling logic is implemented in [src/modeling.py](src/modeling.py). The workflow uses a preprocessing pipeline that imputes missing values and standardizes numeric features before classification.

Two candidate classifiers were evaluated:

1. Logistic Regression
2. Random Forest

The project used a stratified 5-fold cross-validation procedure and selected the best model using mean cross-validation ROC-AUC. The final metrics recorded in [artifacts/metrics.json](artifacts/metrics.json) are:

- Logistic Regression: CV ROC-AUC = 0.9001, test accuracy = 0.8525, precision = 0.8065, recall = 0.8929, ROC-AUC = 0.9578
- Random Forest: CV ROC-AUC = 0.8903, test accuracy = 0.8852, precision = 0.8387, recall = 0.9286, ROC-AUC = 0.9578

The selected model was Logistic Regression because it achieved the highest mean cross-validation ROC-AUC while retaining a strong and interpretable decision boundary. The final model is saved to [artifacts/model.joblib](artifacts/model.joblib).

## 5. Experiment tracking and reproducibility

The project integrates MLflow to log parameters, metrics, and artifacts for every training run. The experiment configuration includes model type, cross-validation settings, random seed, and selected hyperparameters for each model. The output is recorded under the local MLflow store at [mlruns](mlruns).

This setup ensures that:

- model experiments are traceable;
- improvements can be compared against prior runs;
- model artifacts and reports remain attached to each run;
- the workflow is reproducible from the same data and model source.

## 6. Packaging and environment setup

The repository includes all key packaging files:

- [requirements.txt](requirements.txt)
- [requirements-dev.txt](requirements-dev.txt)
- [Dockerfile](Dockerfile)
- [docker-compose.yml](docker-compose.yml)
- [src](src)

The environment can be recreated with a clean Python installation by running the dependency install commands shown in [README.md](README.md). The final model pipeline is serialized in a reusable format using joblib, and the feature preprocessing logic is embedded in the training pipeline so the same transformation steps used during development are preserved for inference.

## 7. Automated testing and CI/CD

The project includes a tested Python workflow with unit tests for the data processing logic and the API interface. These are implemented in [tests/test_data.py](tests/test_data.py), [tests/test_api.py](tests/test_api.py), and [tests/test_modeling.py](tests/test_modeling.py).

The CI workflow is defined in [.github/workflows/ci.yml](.github/workflows/ci.yml). It performs:

- dependency installation;
- Ruff linting;
- Pytest validation;
- dataset generation;
- training pipeline execution;
- artifact upload for downstream inspection.

This provides continuous validation and ensures that changes to the preprocessing logic, feature engineering code, or model training code are caught early.

## 8. API, containerization, and local deployment

The inference API is implemented in [src/api.py](src/api.py). It exposes:

- /health
- /schema
- /predict
- /metrics

The service uses FastAPI and exposes Prometheus-compatible counters and histograms, allowing operational monitoring of prediction requests.

The application is containerized through [Dockerfile](Dockerfile) and can be run locally through Docker Compose using [docker-compose.yml](docker-compose.yml). The Kubernetes deployment definition is located in [k8s/deployment.yaml](k8s/deployment.yaml). This configuration provides the deployment blueprint for a local or cluster-based API rollout.

## 9. Monitoring and logging

The FastAPI service logs each prediction request and exposes the metrics endpoint. This is an appropriate lightweight monitoring pattern for an educational MLOps system. The metrics endpoint provides request counters and latency measurements, which can be consumed by Prometheus and adapted for a visual dashboard if needed.

## 10. Architecture and evidence summary

The system architecture is composed of four main layers:

1. Source data and ETL pipeline
2. Model development and experiment tracking
3. API serving and health monitoring
4. Deployment and CI validation

The repository structure and assets support this workflow across local development, Docker packaging, and Kubernetes deployment.

## 11. Repository and evidence links

- Repository root: [README.md](README.md)
- Architecture notes: [ARCHITECTURE.md](ARCHITECTURE.md)
- Deployment guide: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- Local walkthrough: [LOCAL_DEMO_GUIDE.md](LOCAL_DEMO_GUIDE.md)
- Training metrics: [artifacts/metrics.json](artifacts/metrics.json)
- EDA and model plots: [artifacts/plots](artifacts/plots)
- Proof screenshots: [screenshots](screenshots)

This report reflects the locally verified implementation, the actual artifact files in the repository, and the relevant evidence generated during the assignment workflow.

## 12. Final note

This project demonstrates a practical MLOps workflow for a binary classification task and highlights the importance of reproducibility, validation, monitoring, and containerized deployment. It is intended for academic learning and should not be interpreted as a clinical decision system.
