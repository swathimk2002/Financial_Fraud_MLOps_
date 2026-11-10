
"""Train and compare fraud detection models."""

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing.preprocessor import build_preprocessor


def get_models(random_state: int = 42) -> dict:
    """Return the models to compare."""

    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=1,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def compare_models(
    X_train,
    y_train,
    X_validation,
    y_validation,
    random_state: int = 42,
) -> tuple[dict, dict]:
    """Train models and compare their validation performance."""

    models = get_models(random_state)

    trained_models = {}
    results = {}

    for model_name, model in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", clone(model)),
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_validation)
        probabilities = pipeline.predict_proba(X_validation)[:, 1]

        results[model_name] = {
            "average_precision": average_precision_score(
                y_validation, probabilities
            ),
            "precision": precision_score(
                y_validation, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_validation, predictions, zero_division=0
            ),
            "f1_score": f1_score(
                y_validation, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(
                y_validation, probabilities
            ),
            "confusion_matrix": confusion_matrix(
                y_validation, predictions
            ).tolist(),
        }

        trained_models[model_name] = pipeline

        print(f"{model_name} training completed.")

    return trained_models, results
