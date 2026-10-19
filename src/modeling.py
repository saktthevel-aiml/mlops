"""Train and compare reproducible classification pipelines with MLflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import FEATURES, TARGET, clean_frame, load_uci_heart_disease


def build_preprocessor() -> ColumnTransformer:
    """All inputs are numeric in the selected Cleveland subset."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([("numeric", numeric, FEATURES)], remainder="drop")


def model_candidates(seed: int = 42) -> dict:
    prep = build_preprocessor()
    return {
        "logistic_regression": (
            Pipeline([("preprocess", prep), ("classifier", LogisticRegression(max_iter=2000, random_state=seed))]),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            Pipeline([("preprocess", prep), ("classifier", RandomForestClassifier(random_state=seed, class_weight="balanced"))]),
            {"classifier__n_estimators": [100, 250], "classifier__max_depth": [None, 4, 8]},
        ),
    }


def create_eda_artifacts(frame: pd.DataFrame, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    p = out_dir / "class_balance.png"
    ax = sns.countplot(data=frame, x=TARGET)
    ax.set_title("Heart disease target balance")
    ax.set_xlabel("Target (0 = absent, 1 = present)")
    ax.figure.tight_layout()
    ax.figure.savefig(p, dpi=150)
    plt.close(ax.figure)
    paths.append(p)

    p = out_dir / "feature_histograms.png"
    frame[FEATURES].hist(figsize=(14, 10), bins=18)
    plt.suptitle("Feature distributions")
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
    paths.append(p)

    p = out_dir / "correlation_heatmap.png"
    plt.figure(figsize=(12, 9))
    sns.heatmap(frame.corr(numeric_only=True), cmap="vlag", center=0, annot=False)
    plt.title("Feature and target correlation heatmap")
    plt.tight_layout()
    plt.savefig(p, dpi=150)
    plt.close()
    paths.append(p)
    return paths


def train(frame: pd.DataFrame, model_dir: Path, tracking_uri: str = "./mlruns") -> dict:
    frame = clean_frame(frame)
    X, y = frame[FEATURES], frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("uci-heart-disease-classification")
    artifact_dir = model_dir / "plots"
    eda_paths = create_eda_artifacts(frame, artifact_dir)

    records = []
    best_estimator = None
    best_name = None
    best_cv_auc = -1.0

    for name, (pipeline, grid) in model_candidates().items():
        with mlflow.start_run(run_name=name) as run:
            search = GridSearchCV(
                pipeline, grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            fitted = search.best_estimator_
            pred = fitted.predict(X_test)
            prob = fitted.predict_proba(X_test)[:, 1]
            metrics = {
                "cv_best_roc_auc": float(search.best_score_),
                "test_accuracy": float(accuracy_score(y_test, pred)),
                "test_precision": float(precision_score(y_test, pred, zero_division=0)),
                "test_recall": float(recall_score(y_test, pred, zero_division=0)),
                "test_roc_auc": float(roc_auc_score(y_test, prob)),
            }
            mlflow.log_params(search.best_params_)
            mlflow.log_param("model", name)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_metrics(metrics)
            mlflow.log_text(classification_report(y_test, pred, zero_division=0), "classification_report.txt")
            cm_path = artifact_dir / f"{name}_confusion_matrix.png"
            cm = confusion_matrix(y_test, pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{name}: confusion matrix")
            fig.tight_layout()
            fig.savefig(cm_path, dpi=150)
            plt.close(fig)
            roc_path = artifact_dir / f"{name}_roc.png"
            fig, ax = plt.subplots()
            RocCurveDisplay.from_predictions(y_test, prob, ax=ax)
            ax.set_title(f"{name}: ROC curve")
            fig.tight_layout()
            fig.savefig(roc_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifact(str(roc_path))
            for path in eda_paths:
                mlflow.log_artifact(str(path))
            input_example = X_train.head(5).astype("float64")
            signature = infer_signature(input_example, fitted.predict(input_example))
            mlflow.sklearn.log_model(
                fitted,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )
            record = {
                "run_id": run.info.run_id, "model": name,
                "best_params": search.best_params_, **metrics,
            }
            records.append(record)
            if metrics["cv_best_roc_auc"] > best_cv_auc:
                best_cv_auc = metrics["cv_best_roc_auc"]
                best_estimator, best_name = fitted, name

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    joblib.dump(best_estimator, model_path)
    summary = {
        "selected_model": best_name,
        "selection_metric": "mean cross-validation ROC-AUC",
        "results": records,
        "feature_order": FEATURES,
        "warning": "Educational dataset only; not for clinical use.",
    }
    (model_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/heart_disease.csv")
    parser.add_argument("--model-dir", default="artifacts")
    parser.add_argument("--tracking-uri", default="./mlruns")
    args = parser.parse_args()
    data_path = Path(args.data)
    frame = pd.read_csv(data_path) if data_path.exists() else load_uci_heart_disease()
    train(frame, Path(args.model_dir), args.tracking_uri)


if __name__ == "__main__":
    main()
