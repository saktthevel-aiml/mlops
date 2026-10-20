"""Command-line inference using the saved preprocessing + model pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data import FEATURES


def predict_record(model, record: dict) -> dict:
    missing = [c for c in FEATURES if c not in record]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    row = pd.DataFrame([[record[c] for c in FEATURES]], columns=FEATURES)
    probabilities = model.predict_proba(row)[0]
    prediction = int(model.predict(row)[0])
    return {
        "prediction": prediction,
        "confidence": round(float(max(probabilities)), 5),
        "class_probabilities": {str(i): round(float(p), 5) for i, p in enumerate(probabilities)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict from a JSON record (educational use only).")
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--input", required=True, help="Path to a JSON file with the 13 features")
    args = parser.parse_args()
    model = joblib.load(args.model)
    record = json.loads(Path(args.input).read_text())
    print(json.dumps(predict_record(model, record), indent=2))


if __name__ == "__main__":
    main()
