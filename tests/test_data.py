import pandas as pd
import pytest

from src.data import FEATURES, TARGET, clean_frame


def sample_frame():
    row = {feature: 1 for feature in FEATURES}
    row["age"] = 50
    row[TARGET] = 1
    return pd.DataFrame([row, {**row, TARGET: 0}])


def test_clean_frame_keeps_expected_columns_and_binary_target():
    cleaned = clean_frame(sample_frame())
    assert list(cleaned.columns) == FEATURES + [TARGET]
    assert set(cleaned[TARGET]) == {0, 1}


def test_clean_frame_rejects_missing_feature():
    frame = sample_frame().drop(columns=["age"])
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_frame(frame)


def test_clean_frame_preserves_missing_feature_for_pipeline_imputation():
    frame = sample_frame()
    frame.loc[0, "chol"] = float("nan")
    cleaned = clean_frame(frame)
    assert cleaned["chol"].isna().sum() == 1
