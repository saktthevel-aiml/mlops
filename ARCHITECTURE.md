# Architecture

```mermaid
flowchart TD
  A[UCI Heart Disease dataset] --> B[Data validation and cleaning]
  B --> C[EDA plots and class balance]
  B --> D[Train/test split]
  D --> E[Preprocessing pipeline]
  E --> F1[Logistic Regression + GridSearchCV]
  E --> F2[Random Forest + GridSearchCV]
  F1 --> G[MLflow parameters, metrics, plots]
  F2 --> G
  F1 --> H[Selected joblib pipeline]
  F2 --> H
  H --> I[FastAPI /predict]
  I --> J[Docker image]
  J --> K[Kubernetes Deployment + LoadBalancer Service]
  I --> L[JSON logs + Prometheus metrics]
  M[GitHub Actions] --> N[Ruff + pytest + training + Docker build]
```

The CI workflow verifies quality and creates artifacts; the Kubernetes deployment is intentionally a manual release step so an image registry and cluster credentials are not assumed.
