
"""Select the best model using validation metrics."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelSelectionResult:
    """Store the selected model and its evaluation score."""

    model_name: str
    model: Any
    metric: str
    score: float


def select_best_model(
    trained_models: dict,
    results: dict,
    metric: str = "average_precision",
) -> ModelSelectionResult:
    """Select the model with the highest validation metric."""

    if not trained_models:
        raise ValueError("No trained models were provided.")

    if metric not in {
        "average_precision",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    }:
        raise ValueError(f"Unsupported selection metric: {metric}")

    missing_results = set(trained_models) - set(results)

    if missing_results:
        raise ValueError(
            f"Missing evaluation results for: {sorted(missing_results)}"
        )

    for model_name in trained_models:
        if metric not in results[model_name]:
            raise ValueError(
                f"Metric '{metric}' is missing for model '{model_name}'."
            )

    best_model_name = max(
        trained_models,
        key=lambda name: results[name][metric],
    )

    best_score = float(results[best_model_name][metric])

    return ModelSelectionResult(
        model_name=best_model_name,
        model=trained_models[best_model_name],
        metric=metric,
        score=best_score,
    )
