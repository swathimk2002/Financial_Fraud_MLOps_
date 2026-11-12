import os
from pathlib import Path
from numbers import Real

import mlflow
import mlflow.sklearn
import yaml
from mlflow.tracking import MlflowClient


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "mlflow.yaml"


def load_config():
    """Load MLflow settings and resolve environment variables."""

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    tracking = config["tracking"]

    tracking["backend_store_uri"] = os.getenv(
        "MLFLOW_TRACKING_URI",
        "sqlite:///mlflow.db",
    )

    tracking["artifact_root"] = os.getenv(
        "MLFLOW_ARTIFACT_ROOT",
        "file:///D:/Financial_Fraud_MLOps/mlartifacts",
    )

    return config


def initialize_mlflow():
    """Connect to MLflow and create the experiment."""

    config = load_config()

    tracking_uri = config["tracking"]["backend_store_uri"]
    artifact_root = config["tracking"]["artifact_root"]
    experiment_name = config["experiment"]["name"]

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_root,
        )

    mlflow.set_experiment(experiment_name)

    return MlflowClient()


def log_candidate_runs(trained_models, results, selection_metric):
    """Log each trained model and its validation results."""

    initialize_mlflow()
    run_details = {}

    for model_name, pipeline in trained_models.items():

        with mlflow.start_run(run_name=model_name) as run:

            mlflow.set_tags({
                "run_type": "candidate_model",
                "model_name": model_name,
                "selection_metric": selection_metric,
            })

            model_parameters = pipeline.named_steps["model"].get_params(
                deep=False
            )

            parameters = {
                key: str(value)[:500]
                for key, value in model_parameters.items()
                if isinstance(value, (str, int, float, bool))
                or value is None
            }

            mlflow.log_params(parameters)

            metrics = results[model_name]

            scalar_metrics = {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, Real)
            }

            mlflow.log_metrics(scalar_metrics)

            mlflow.log_dict(
                {"confusion_matrix": metrics["confusion_matrix"]},
                "confusion_matrix.json",
            )

            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="cloudpickle",
            )

            run_details[model_name] = {
                "run_id": run.info.run_id,
                "model_uri": model_info.model_uri,
            }

            print(f"MLflow run logged: {model_name}")

    return run_details


def register_selected_model(model_uri):
    """Register the selected model and assign its champion alias."""

    config = load_config()
    client = initialize_mlflow()

    model_name = config["model_registry"]["registered_model_name"]

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=str(model_version.version),
    )

    print(f"Registered model: {model_name}")
    print(f"Model version: {model_version.version}")
    print("Model alias: champion")

    return model_version


def log_final_evaluation(report_path, metadata_path, figures_dir):
    """Log final evaluation results and graphs to MLflow."""

    import json

    initialize_mlflow()

    with open(report_path, "r", encoding="utf-8") as file:
        report = json.load(file)

    with mlflow.start_run(run_name="final_model_evaluation"):

        mlflow.set_tags({
            "run_type": "final_evaluation",
            "model_name": report["model_name"],
            "evaluation_type": "validation_and_test",
        })

        mlflow.log_params({
            "model_name": report["model_name"],
            "model_version": str(report["model_version"]),
            "classification_threshold": str(
                report["classification_threshold"]
            ),
        })

        for split_name in ("validation_metrics_at_threshold", "test_metrics"):
            metrics = report[split_name]

            mlflow.log_metrics({
                f"{split_name}_{name}": float(value)
                for name, value in metrics.items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
            })

        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(metadata_path))

        graph_names = [
            "roc_curve.png",
            "precision_recall_curve.png",
            "confusion_matrix_updated.png",
            "threshold_comparison.png",
        ]

        for graph_name in graph_names:
            graph_path = Path(figures_dir) / graph_name

            if not graph_path.is_file():
                raise FileNotFoundError(f"Graph not found: {graph_path}")

            mlflow.log_artifact(str(graph_path), artifact_path="figures")

        print("Final evaluation and all four graphs logged to MLflow.")
