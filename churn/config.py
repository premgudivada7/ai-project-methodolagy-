"""Project-wide configuration: paths, column names and MLflow settings.

All scripts import their settings from this module so that the project has a
single source of truth. Values can be overridden with environment variables
where it makes sense (e.g. the MLflow tracking URI).
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (cookiecutter-data-science layout)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SHAP_DIR = FIGURES_DIR / "shap"

RAW_FILE_CANDIDATES = (
    "E Commerce Dataset.xlsx",
    "E_Commerce_Dataset.xlsx",
    "ecommerce_churn.xlsx",
    "ecommerce_churn.csv",
)
RAW_SHEET_NAME = "E Comm"

CLEAN_FILE = INTERIM_DIR / "churn_clean.csv"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
LOCAL_MODEL_FILE = MODELS_DIR / "churn_model.joblib"

# --------------------------------------------------------------------------
# Dataset schema
# --------------------------------------------------------------------------
ID_COL = "CustomerID"
TARGET = "Churn"

NUMERIC_FEATURES = [
    "Tenure",
    "CityTier",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
]

CATEGORICAL_FEATURES = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]

# Features created in churn.features.add_features
ENGINEERED_FEATURES = [
    "CashbackPerOrder",
    "CouponPerOrder",
    "IsNewCustomer",
]

# Gender is kept in the raw data but excluded from the model (fairness / GDPR
# data minimisation). It is still available for fairness checks.
EXCLUDED_FEATURES = ["Gender"]

# Harmonisation of inconsistent category labels found during EDA
CATEGORY_MAPPING = {
    "PreferredLoginDevice": {"Phone": "Mobile Phone"},
    "PreferredPaymentMode": {"CC": "Credit Card", "COD": "Cash on Delivery"},
    "PreferedOrderCat": {"Mobile": "Mobile Phone"},
}

# --------------------------------------------------------------------------
# Modelling
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --------------------------------------------------------------------------
# MLflow
# --------------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
MLFLOW_ARTIFACT_LOCATION = (PROJECT_ROOT / "mlruns").as_uri()
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "retailgenius-churn")
REGISTERED_MODEL_NAME = "retailgenius-churn-model"
CHAMPION_ALIAS = "champion"


def model_features() -> list:
    """Return the ordered list of input columns used by the model.

    Returns:
        list: numeric + engineered + categorical features, without the
        excluded (sensitive) attributes.
    """
    cols = NUMERIC_FEATURES + ENGINEERED_FEATURES + CATEGORICAL_FEATURES
    return [c for c in cols if c not in EXCLUDED_FEATURES]
