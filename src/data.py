"""Download and clean the UCI Heart Disease Cleveland dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET = "target"


def load_uci_heart_disease() -> pd.DataFrame:
    """Fetch UCI dataset 45 and return 13 features plus binary target."""
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=45)
    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()
    # UCI column labels are typically lower-case; normalize defensively.
    X.columns = [str(c).strip().lower() for c in X.columns]
    y.columns = [str(c).strip().lower() for c in y.columns]
    missing = sorted(set(FEATURES) - set(X.columns))
    if missing:
        raise ValueError(f"Unexpected UCI schema; missing features: {missing}")
    X = X[FEATURES].copy()
    raw_target = pd.to_numeric(y.iloc[:, 0], errors="coerce")
    if raw_target.isna().all():
        raise ValueError("Target column could not be converted to numeric.")
    frame = X.copy()
    frame[TARGET] = (raw_target > 0).astype("int64")
    # '?' strings in some mirrors become NaN on numeric conversion.
    for col in FEATURES:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, normalize numeric types, and retain missing values for imputation."""
    expected = FEATURES + [TARGET]
    missing = [c for c in expected if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    result = frame[expected].copy()
    for col in FEATURES:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    result[TARGET] = pd.to_numeric(result[TARGET], errors="coerce")
    result = result.dropna(subset=[TARGET])
    result[TARGET] = result[TARGET].astype(int)
    if not set(result[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only binary values 0 and 1.")
    if result.empty:
        raise ValueError("No usable rows remain after cleaning.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/heart_disease.csv")
    args = parser.parse_args()
    frame = clean_frame(load_uci_heart_disease())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame)} rows to {output}")
    print(frame[TARGET].value_counts(dropna=False).sort_index())


if __name__ == "__main__":
    main()
