
from pathlib import Path

import pandas as pd
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "transaction_features.parquet"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "monitoring"


# Pre-transaction model features only
FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "amount_to_orig_balance_ratio",
    "amount_to_dest_balance_ratio",
]


def run_data_drift_report():
    # Check that the source dataset exists
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_PATH}"
        )

    # Load the engineered feature dataset
    df = pd.read_parquet(DATA_PATH)

    # Validate required columns
    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required feature columns: {missing_columns}"
        )

    # Create chronological reference and comparison periods
    reference_df = df.loc[
        df["step"].between(1, 368),
        FEATURE_COLUMNS,
    ].copy()

    comparison_df = df.loc[
        df["step"].between(369, 736),
        FEATURE_COLUMNS,
    ].copy()

    # Validate both periods
    if reference_df.empty or comparison_df.empty:
        raise ValueError(
            "Reference or comparison dataset is empty."
        )

    if reference_df.isnull().any().any():
        raise ValueError("Reference dataset contains missing values.")

    if comparison_df.isnull().any().any():
        raise ValueError("Comparison dataset contains missing values.")

    # Convert pandas DataFrames to Evidently datasets
    reference_data = Dataset.from_pandas(reference_df)
    comparison_data = Dataset.from_pandas(comparison_df)

    # Configure feature drift report
    report = Report(
        metrics=[
            DataDriftPreset(columns=FEATURE_COLUMNS)
        ],
        metadata={
            "project": "Financial Fraud Detection MLOps",
            "monitoring_type": "Historical feature drift demonstration",
            "reference_period": "PaySim steps 1-368",
            "comparison_period": "PaySim steps 369-736",
        },
    )

    # Run comparison
    snapshot = report.run(
        reference_data=reference_data,
        current_data=comparison_data,
    )

    # Create output directory if necessary
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Save reports
    html_path = REPORT_DIR / "historical_data_drift.html"
    json_path = REPORT_DIR / "historical_data_drift.json"

    snapshot.save_html(str(html_path))
    snapshot.save_json(str(json_path))

    print("\nHistorical feature drift report generated.")
    print(f"Reference rows: {len(reference_df):,}")
    print(f"Comparison rows: {len(comparison_df):,}")
    print(f"Features monitored: {len(FEATURE_COLUMNS)}")
    print(f"HTML report: {html_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    run_data_drift_report()
