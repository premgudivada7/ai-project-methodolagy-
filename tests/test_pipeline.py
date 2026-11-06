"""Unit tests for data preparation, features, training helpers and inference."""

import numpy as np
import pandas as pd
import pytest

from churn import config
from churn.dataset import clean, split, validate_schema
from churn.features import add_features, build_preprocessor, split_xy
from churn.predict import prepare, score


def test_validate_schema_detects_missing_column(raw_df):
    validate_schema(raw_df)
    with pytest.raises(ValueError):
        validate_schema(raw_df.drop(columns=["Tenure"]))


def test_clean_harmonises_categories_and_duplicates(raw_df):
    dup = raw_df.iloc[[0]].assign(CustomerID=99999)
    df = clean(pd.concat([raw_df, dup], ignore_index=True))
    assert len(df) == len(raw_df)  # copied customer removed
    assert "Phone" not in set(df["PreferredLoginDevice"])
    assert "CC" not in set(df["PreferredPaymentMode"])
    assert "Mobile" not in set(df["PreferedOrderCat"])
    assert df[config.ID_COL].is_unique


def test_split_is_stratified(raw_df):
    train, test = split(clean(raw_df))
    assert abs(train[config.TARGET].mean() - test[config.TARGET].mean()) < 0.05


def test_add_features_no_division_by_zero(raw_df):
    df = add_features(clean(raw_df))
    assert set(config.ENGINEERED_FEATURES) <= set(df.columns)
    assert not np.isinf(df["CashbackPerOrder"]).any()
    assert df["CouponPerOrder"].max() <= 1


def test_gender_is_excluded_from_model_inputs():
    assert "Gender" not in config.model_features()


def test_preprocessor_and_model_end_to_end(raw_df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    df = add_features(clean(raw_df))
    X, y = split_xy(df)
    pipe = Pipeline(
        [("preprocess", build_preprocessor()), ("model", RandomForestClassifier(n_estimators=10))]
    )
    pipe.fit(X, y)
    out = score(pipe, prepare(raw_df.drop(columns=[config.TARGET])))
    assert len(out) == len(raw_df)
    assert out["churn_probability"].between(0, 1).all()
    assert set(out["risk_segment"].astype(str)) <= {"Low", "Medium", "High"}


def test_metrics():
    pytest.importorskip("mlflow")
    from churn.train import evaluate, lift_at

    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = evaluate(y, p)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert lift_at(y, p, 0.25) == pytest.approx(2.0)
