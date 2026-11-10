
"""Run and display the baseline model comparison."""

from src.ingestion.loader import load_file
from src.features.engineer import engineer_features
from src.preprocessing.splitter import split_dataset
from src.models.trainer import compare_models


def main():
    """Load data, engineer features, split data and compare models."""

    df = load_file("data/samples/paysim_sample_200k.csv")

    # Create features available before transaction processing.
    df = engineer_features(df)

    # Split the engineered dataset.
    split = split_dataset(df)

    print("Starting pre-transaction model training...")
    print("This may take a few minutes.\n")

    trained_models, results = compare_models(
        split.X_train,
        split.y_train,
        split.X_validation,
        split.y_validation,
    )

    print("\nPRE-TRANSACTION MODEL COMPARISON RESULTS")
    print("=" * 60)

    for name, metrics in results.items():
        print(f"\n{name}")
        print("-" * 40)
        print(f"Average Precision: {metrics['average_precision']:.4f}")
        print(f"Precision:         {metrics['precision']:.4f}")
        print(f"Recall:            {metrics['recall']:.4f}")
        print(f"F1-score:          {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
        print(f"Confusion Matrix:  {metrics['confusion_matrix']}")

    print("\nPRE-TRANSACTION MODEL COMPARISON COMPLETED.")


if __name__ == "__main__":
    main()
