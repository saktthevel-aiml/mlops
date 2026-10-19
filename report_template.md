# MLOps Assignment 01: End-to-End Heart Disease Classification

**Student:** [Your name]  
**Course:** AIMLCZG523 — MLOps  
**Repository:** [GitHub URL]  
**Deployment URL/access instructions:** [URL or local instructions]  
**Date:** [Date]

> Complete this report using your own experiment outputs and screenshots. Do not leave placeholders in the final submission.

## 1. Executive summary
Describe the task, solution, selected model, measured held-out metrics, deployment state, and limitations. Use only actual results from `artifacts/metrics.json`.

## 2. Problem statement and architecture
Explain binary classification and the end-to-end flow. Insert your architecture diagram and explain each component.

## 3. Data acquisition and data quality
Identify the UCI dataset and acquisition method. State row/column counts from your actual run, target conversion, missing-value counts, and any limitations. Explain how missing values are handled without leaking test data into training.

## 4. Exploratory data analysis
Include the class-balance chart, feature histograms, and correlation heatmap. Discuss at least three observations that you personally verified from the plots.

## 5. Feature engineering and model development
Describe feature order, median imputation, standardization, train/test split, stratification, cross-validation, random seed, candidate models, and hyperparameter grids. Explain why these choices are suitable.

## 6. Experiment tracking and model selection
Insert an MLflow screenshot. Compare Logistic Regression and Random Forest using the recorded cross-validation ROC-AUC and held-out accuracy, precision, recall, and ROC-AUC. Explain the selection rule and any limitations of comparing on a small test set.

## 7. Reproducibility and packaging
Explain saved `joblib` pipeline, requirements file, model path, and clean-environment execution. Include relevant commands.

## 8. Automated tests and CI/CD
Describe test cases, Ruff linting, training, Docker build, workflow triggers, failure behavior, and artifact retention. Insert actual successful/failed Actions screenshots.

## 9. API, containerization and deployment
Document `/health`, `/schema`, `/predict`, `/metrics`; show a request and response from your run. Explain Docker image creation and the deployment target. Include Kubernetes resource details and actual service URL or local access instructions.

## 10. Monitoring, security and limitations
Explain request logs, Prometheus metrics, dashboard, and what they measure. Discuss privacy, no patient identifiers, TLS/authentication for public use, model/data limitations, dataset bias, and why this is not a medical diagnostic system.

## Conclusion
Summarize what works, what you learned, and the next improvements.

## References
- UCI Machine Learning Repository: Heart Disease dataset, https://archive.ics.uci.edu/dataset/45/heart+disease
- MLflow documentation: https://mlflow.org/docs/latest/
- FastAPI documentation: https://fastapi.tiangolo.com/
- scikit-learn documentation: https://scikit-learn.org/stable/
- GitHub Actions documentation: https://docs.github.com/actions
- Kubernetes documentation: https://kubernetes.io/docs/
