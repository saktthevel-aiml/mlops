from src.modeling import build_preprocessor, model_candidates


def test_preprocessor_and_candidates_are_defined():
    assert build_preprocessor() is not None
    candidates = model_candidates()
    assert set(candidates) == {"logistic_regression", "random_forest"}
    for pipeline, params in candidates.values():
        assert "preprocess" in pipeline.named_steps
        assert params
