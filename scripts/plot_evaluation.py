
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    precision_score,
    recall_score,
    f1_score,
)

from src.ingestion.loader import load_file
from src.features.engineer import engineer_features
from src.preprocessing.splitter import split_dataset


# Project paths
ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Load selected model and metadata
    model = joblib.load(ARTIFACTS_DIR / "selected_model.joblib")

    with open(ARTIFACTS_DIR / "model_metadata.json", "r") as file:
        metadata = json.load(file)

    threshold = metadata["classification_threshold"]

    # Load and prepare dataset
    data_path = ROOT_DIR / "data" / "samples" / "paysim_sample_200k.csv"
    df = engineer_features(load_file(data_path))
    split = split_dataset(df)

    # Generate prediction probabilities
    validation_probabilities = model.predict_proba(
        split.X_validation
    )[:, 1]

    test_probabilities = model.predict_proba(
        split.X_test
    )[:, 1]

    # 1. ROC Curve - validation data
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(
        split.y_validation,
        validation_probabilities,
        ax=ax
    )
    ax.set_title("ROC Curve - Validation Data")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curve.png", dpi=300)
    plt.close(fig)

    # 2. Precision-Recall Curve - validation data
    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(
        split.y_validation,
        validation_probabilities,
        ax=ax
    )
    ax.set_title("Precision-Recall Curve - Validation Data")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "precision_recall_curve.png", dpi=300)
    plt.close(fig)

    # 3. Confusion Matrix - final test data
    test_predictions = (test_probabilities >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay.from_predictions(
        split.y_test,
        test_predictions,
        display_labels=["Legitimate", "Fraud"],
        values_format="d",
        ax=ax
    )
    ax.set_title(f"Test Confusion Matrix (Threshold = {threshold})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix_updated.png", dpi=300)
    plt.close(fig)

    # 4. Threshold comparison - validation data only
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    precisions = []
    recalls = []
    f1_scores = []

    for value in thresholds:
        predictions = (
            validation_probabilities >= value
        ).astype(int)

        precisions.append(
            precision_score(split.y_validation, predictions, zero_division=0)
        )
        recalls.append(
            recall_score(split.y_validation, predictions)
        )
        f1_scores.append(
            f1_score(split.y_validation, predictions)
        )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, precisions, marker="o", label="Precision")
    ax.plot(thresholds, recalls, marker="o", label="Recall")
    ax.plot(thresholds, f1_scores, marker="o", label="F1-score")

    ax.axvline(
        x=threshold,
        linestyle="--",
        label=f"Selected threshold ({threshold})"
    )

    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Comparison - Validation Data")
    ax.set_xticks(thresholds)
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "threshold_comparison.png", dpi=300)
    plt.close(fig)

    print("Evaluation graphs generated successfully.")
    print("\nGenerated files:")

    for figure in sorted(FIGURES_DIR.glob("*.png")):
        print(f"{figure.name} - {figure.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
