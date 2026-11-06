"""Feature engineering step.

Creates business features and defines the preprocessing pipeline
(imputation, encoding, scaling) that is fitted inside the model pipeline.

Usage::

    python -m churn.features
"""

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn import config

logger = logging.getLogger(__name__)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features.

    * ``CashbackPerOrder`` – cashback amount divided by number of orders,
    * ``CouponPerOrder`` – share of orders that used a coupon,
    * ``IsNewCustomer`` – 1 if tenure is 1 month or less (new customers
      churn much more).

    Missing inputs stay missing (they are imputed later in the pipeline).

    Args:
        df: cleaned dataframe.

    Returns:
        A copy of the dataframe with the new columns.
    """
    df = df.copy()
    orders = df["OrderCount"].astype(float)
    safe_orders = orders.where(orders > 0, np.nan)
    df["CashbackPerOrder"] = df["CashbackAmount"] / safe_orders
    df["CouponPerOrder"] = (df["CouponUsed"] / safe_orders).clip(upper=1)
    df["IsNewCustomer"] = np.where(df["Tenure"].isna(), np.nan, (df["Tenure"] <= 1).astype(float))
    return df


def numeric_columns() -> list:
    """Numeric model inputs (raw + engineered)."""
    return [
        c
        for c in config.NUMERIC_FEATURES + config.ENGINEERED_FEATURES
        if c not in config.EXCLUDED_FEATURES
    ]


def categorical_columns() -> list:
    """Categorical model inputs."""
    return [c for c in config.CATEGORICAL_FEATURES if c not in config.EXCLUDED_FEATURES]


def build_preprocessor(scale: bool = False) -> ColumnTransformer:
    """Build the preprocessing transformer.

    Args:
        scale: add a ``StandardScaler`` on numeric columns (useful for
            linear models; useless for tree models).

    Returns:
        An unfitted :class:`~sklearn.compose.ColumnTransformer`.
    """
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(num_steps)
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, numeric_columns()),
            ("cat", categorical, categorical_columns()),
        ],
        verbose_feature_names_out=False,
    )


def split_xy(df: pd.DataFrame):
    """Separate model inputs and target.

    Numeric inputs are cast to ``float`` so that the MLflow model signature
    accepts missing values at inference time (integer columns cannot hold
    NaN).

    Args:
        df: dataframe containing features and the target.

    Returns:
        tuple: ``(X, y)``.
    """
    X = df[config.model_features()].copy()
    num = numeric_columns()
    X[num] = X[num].astype(float)
    return X, df[config.TARGET].astype(int)


def run() -> None:
    """Apply feature engineering to the interim train / test files."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, out in (("train", config.TRAIN_FILE), ("test", config.TEST_FILE)):
        df = pd.read_csv(config.INTERIM_DIR / f"{name}_raw.csv")
        df = add_features(df)
        cols = [config.ID_COL] + config.model_features() + [config.TARGET]
        df[cols].to_csv(out, index=False)
        logger.info("Wrote %s (%d rows, %d features)", out.name, len(df), len(cols) - 2)


def main() -> None:
    """Command-line entry point."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run()


if __name__ == "__main__":
    main()
