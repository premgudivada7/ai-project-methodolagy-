"""Shared test fixtures: a small synthetic dataset with the raw schema."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Return 200 fake customers with the same columns as the Kaggle file."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "CustomerID": np.arange(50001, 50001 + n),
            "Churn": rng.integers(0, 2, n),
            "Tenure": rng.integers(0, 30, n).astype(float),
            "PreferredLoginDevice": rng.choice(["Mobile Phone", "Phone", "Computer"], n),
            "CityTier": rng.integers(1, 4, n),
            "WarehouseToHome": rng.integers(5, 40, n).astype(float),
            "PreferredPaymentMode": rng.choice(["Debit Card", "CC", "COD", "UPI"], n),
            "Gender": rng.choice(["Male", "Female"], n),
            "HourSpendOnApp": rng.integers(0, 5, n).astype(float),
            "NumberOfDeviceRegistered": rng.integers(1, 6, n),
            "PreferedOrderCat": rng.choice(["Mobile", "Fashion", "Grocery"], n),
            "SatisfactionScore": rng.integers(1, 6, n),
            "MaritalStatus": rng.choice(["Single", "Married", "Divorced"], n),
            "NumberOfAddress": rng.integers(1, 10, n),
            "Complain": rng.integers(0, 2, n),
            "OrderAmountHikeFromlastYear": rng.integers(11, 26, n).astype(float),
            "CouponUsed": rng.integers(0, 5, n).astype(float),
            "OrderCount": rng.integers(0, 10, n).astype(float),
            "DaySinceLastOrder": rng.integers(0, 20, n).astype(float),
            "CashbackAmount": rng.uniform(100, 300, n).round(2),
        }
    )
    df.loc[::17, "Tenure"] = np.nan
    return df
