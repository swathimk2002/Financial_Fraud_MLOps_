
"""Load the saved model and evaluate its final performance."""

import json
from pathlib import Path

import joblib
import numpy as np
from src.mlops.mlflow_manager import log_final_evaluation
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.engineer import engineer_features
from src.ingestion.loader import load_file
from src.preprocessing.splitter import split_dataset


ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "selected_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"
REPORT_PATH = ARTIFACTS_DIR / "evaluation_report.json"


def calculate_metrics(y_true, probabilities, threshold):
    """Calculate classification metrics at a given threshold."""

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def main():
    """Evaluate the saved pipeline on validation and test data."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")

    print("STEP 1: Loading saved model and metadata...")

    model = joblib.load(MODEL_PATH)

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    threshold = float(metadata["classification_threshold"])

    print(f"Loaded model: {metadata['model_name']}")
    print(f"Classification threshold: {threshold}")

    print("\nSTEP 2: Loading and preparing dataset...")

    df = load_file("data/samples/paysim_sample_200k.csv")
    df = engineer_features(df)

    split = split_dataset(
        df,
        random_state=metadata["random_state"],
    )

    print("\nSTEP 3: Generating validation probabilities...")

    validation_probabilities = model.predict_proba(
        split.X_validation
    )[:, 1]

    validation_metrics = calculate_metrics(
        split.y_validation,
        validation_probabilities,
        threshold,
    )

    print("\nVALIDATION METRICS AT SELECTED THRESHOLD")
    print("=" * 50)

    for name, value in validation_metrics.items():
        print(f"{name}: {value}")

    # Correct the metadata so its threshold and metrics agree.
    metadata["validation_metrics_at_threshold"] = validation_metrics

    print("\nSTEP 4: Evaluating untouched test set...")

    test_probabilities = model.predict_proba(
        split.X_test
    )[:, 1]

    test_metrics = calculate_metrics(
        split.y_test,
        test_probabilities,
        threshold,
    )

    print("\nFINAL TEST RESULTS")
    print("=" * 50)

    for name, value in test_metrics.items():
        print(f"{name}: {value}")

    # Record final evaluation separately from validation results.
    metadata["test_metrics"] = test_metrics
    metadata["final_test_evaluation_completed"] = True

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    report = {
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "selection_metric": metadata["selection_metric"],
        "selection_score": metadata["selection_score"],
        "classification_threshold": threshold,
        "validation_metrics_at_threshold": validation_metrics,
        "test_metrics": test_metrics,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nEVALUATION COMPLETED.")
    print(f"Updated metadata: {METADATA_PATH}")
    print(f"Evaluation report: {REPORT_PATH}")

    log_final_evaluation(
    REPORT_PATH,
    METADATA_PATH,
    Path("reports/figures"),
    )
if __name__ == "__main__":
    main()
