
"""Tests for model selection."""

import pytest

from src.model_selection.selector import select_best_model


def test_select_best_model():
    models = {
        "Logistic Regression": object(),
        "Random Forest": object(),
        "XGBoost": object(),
    }

    results = {
        "Logistic Regression": {"average_precision": 0.2733},
        "Random Forest": {"average_precision": 0.8629},
        "XGBoost": {"average_precision": 0.9455},
    }

    selected = select_best_model(models, results)

    assert selected.model_name == "XGBoost"
    assert selected.metric == "average_precision"
    assert selected.score == pytest.approx(0.9455)
    assert selected.model is models["XGBoost"]


def test_select_best_model_with_different_metric():
    models = {
        "Random Forest": object(),
        "XGBoost": object(),
    }

    results = {
        "Random Forest": {"recall": 0.70},
        "XGBoost": {"recall": 0.875},
    }

    selected = select_best_model(
        models,
        results,
        metric="recall",
    )

    assert selected.model_name == "XGBoost"
    assert selected.score == pytest.approx(0.875)


def test_select_best_model_rejects_empty_models():
    with pytest.raises(ValueError, match="No trained models"):
        select_best_model({}, {})


def test_select_best_model_rejects_unsupported_metric():
    with pytest.raises(ValueError, match="Unsupported selection metric"):
        select_best_model(
            {"XGBoost": object()},
            {"XGBoost": {"accuracy": 0.99}},
            metric="accuracy",
        )


def test_select_best_model_rejects_missing_results():
    with pytest.raises(ValueError, match="Missing evaluation results"):
        select_best_model(
            {"XGBoost": object()},
            {},
        )
