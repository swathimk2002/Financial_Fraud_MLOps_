
"""Train, select and save the best fraud detection model."""

import json
import os
from pathlib import Path

import joblib
from dotenv import load_dotenv

from src.features.engineer import engineer_features
from src.ingestion.loader import load_file
from src.model_selection.selector import select_best_model
from src.models.trainer import compare_models
from src.preprocessing.splitter import split_dataset


# Load project environment variables.
load_dotenv()

# Model configuration.
SELECTION_METRIC = os.getenv(
    "MODEL_SELECTION_METRIC",
    "average_precision",
)

CLASSIFICATION_THRESHOLD = 0.60
RANDOM_STATE = 42

# Output locations.
ARTIFACTS_DIR = Path("artifacts")
MODEL_PATH = ARTIFACTS_DIR / "selected_model.joblib"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"


def main():
    """Train models, select the best candidate and save artifacts."""

    print("STEP 1: Loading dataset...")

    df = load_file("data/samples/paysim_sample_200k.csv")

    print(f"Dataset shape: {df.shape}")

    print("\nSTEP 2: Engineering pre-transaction features...")

    df = engineer_features(df)

    print("\nSTEP 3: Splitting dataset...")

    split = split_dataset(
        df,
        random_state=RANDOM_STATE,
    )

    print(f"Training rows: {len(split.X_train)}")
    print(f"Validation rows: {len(split.X_validation)}")
    print(f"Testing rows: {len(split.X_test)}")

    print("\nSTEP 4: Training and comparing models...")

    trained_models, results = compare_models(
        split.X_train,
        split.y_train,
        split.X_validation,
        split.y_validation,
        random_state=RANDOM_STATE,
    )

    print("\nSTEP 5: Selecting the best model...")

    selected = select_best_model(
        trained_models,
        results,
        metric=SELECTION_METRIC,
    )

    print(f"Selected model: {selected.model_name}")
    print(f"Selection metric: {selected.metric}")
    print(f"Validation score: {selected.score:.4f}")

    print("\nSTEP 6: Saving model and metadata...")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save the complete fitted preprocessing and model pipeline.
    joblib.dump(selected.model, MODEL_PATH)

    # Save information needed to identify and use the model.
    metadata = {
        "model_name": selected.model_name,
        "model_version": "1.0.0",
        "task": "binary_classification",
        "prediction_objective": "pre_transaction_fraud_detection",
        "selection_metric": selected.metric,
        "selection_score": selected.score,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "random_state": RANDOM_STATE,
        "target_column": "isFraud",
        "feature_columns": split.X_train.columns.tolist(),
        "validation_metrics": results[selected.model_name],
        "model_file": MODEL_PATH.name,
    }

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    print("\nMODEL SAVING COMPLETED.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")


if __name__ == "__main__":
    main()
