
"""Analyze classification thresholds for the pre-transaction XGBoost model."""

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.ingestion.loader import load_file
from src.features.engineer import engineer_features
from src.preprocessing.splitter import split_dataset
from src.preprocessing.preprocessor import build_preprocessor
from src.models.trainer import get_models


def main():
    """Train XGBoost and evaluate thresholds on validation data."""

    df = load_file("data/samples/paysim_sample_200k.csv")
    df = engineer_features(df)
    split = split_dataset(df)

    model = get_models()["XGBoost"]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )

    print("Training XGBoost for threshold analysis...")
    pipeline.fit(split.X_train, split.y_train)

    probabilities = pipeline.predict_proba(split.X_validation)[:, 1]

    print(
        "\nValidation Average Precision:",
        f"{average_precision_score(split.y_validation, probabilities):.4f}",
    )

    print("\nTHRESHOLD COMPARISON")
    print("=" * 85)
    print(
        f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}"
        f"{'F1-score':<12}{'TP':<8}{'FP':<8}{'FN':<8}"
    )
    print("-" * 85)

    for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:

        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            split.y_validation, predictions, zero_division=0
        )
        recall = recall_score(
            split.y_validation, predictions, zero_division=0
        )
        f1 = f1_score(
            split.y_validation, predictions, zero_division=0
        )

        tn, fp, fn, tp = confusion_matrix(
            split.y_validation, predictions, labels=[0, 1]
        ).ravel()

        print(
            f"{threshold:<12.2f}{precision:<12.4f}{recall:<12.4f}"
            f"{f1:<12.4f}{tp:<8}{fp:<8}{fn:<8}"
        )

    print("\nTHRESHOLD ANALYSIS COMPLETED.")
    print("The test set was not used.")


if __name__ == "__main__":
    main()
