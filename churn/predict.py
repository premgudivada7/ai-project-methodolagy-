"""Inference step: score customers with the trained model.

The model is loaded either from the MLflow Model Registry (default:
``models:/retailgenius-churn-model@champion``) or from a local ``.joblib``
file.

Usage::

    python -m churn.predict --input data/interim/test_raw.csv --output reports/predictions.csv
    python -m churn.predict --model-path models/rf_300_d12.joblib --input new_customers.csv
"""

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churn import config
from churn.dataset import clean
from churn.features import add_features, split_xy

logger = logging.getLogger(__name__)

RISK_BINS = [0.0, 0.3, 0.6, 1.0]
RISK_LABELS = ["Low", "Medium", "High"]


def load_model(model_uri=None, model_path=None):
    """Load a fitted pipeline.

    Args:
        model_uri: MLflow model URI (e.g. ``models:/name@champion``).
        model_path: path to a local joblib file (takes precedence).

    Returns:
        An object exposing ``predict_proba``.
    """
    if model_path:
        return joblib.load(model_path)
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    uri = model_uri or f"models:/{config.REGISTERED_MODEL_NAME}@{config.CHAMPION_ALIAS}"
    logger.info("Loading model %s", uri)
    return mlflow.sklearn.load_model(uri)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same cleaning and feature engineering as in training.

    Args:
        df: raw customer records (same columns as the raw dataset, target
            optional).

    Returns:
        Dataframe with the model input columns.
    """
    df = df.copy()
    if config.TARGET not in df:
        df[config.TARGET] = 0  # placeholder so that clean() can run
    df = add_features(clean(df))
    return df


def score(model, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Score customers.

    Args:
        model: fitted pipeline.
        df: prepared dataframe (output of :func:`prepare`).
        threshold: decision threshold for the churn flag.

    Returns:
        Dataframe with ``CustomerID``, ``churn_probability``,
        ``churn_prediction`` and ``risk_segment``.
    """
    X, _ = split_xy(df)
    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame(
        {
            config.ID_COL: df[config.ID_COL].values,
            "churn_probability": np.round(proba, 4),
            "churn_prediction": (proba >= threshold).astype(int),
            "risk_segment": pd.cut(proba, RISK_BINS, labels=RISK_LABELS, include_lowest=True),
        }
    )
    return out.sort_values("churn_probability", ascending=False)


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", default=str(config.INTERIM_DIR / "test_raw.csv"))
    parser.add_argument("--output", default=str(config.REPORTS_DIR / "predictions.csv"))
    parser.add_argument("--model-uri", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    model = load_model(args.model_uri, args.model_path)
    raw = pd.read_csv(args.input)
    preds = score(model, prepare(raw), args.threshold)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.output, index=False)
    logger.info(
        "Scored %d customers -> %s | segments: %s",
        len(preds),
        args.output,
        preds["risk_segment"].value_counts().to_dict(),
    )


if __name__ == "__main__":
    main()
