"""RetailGenius customer churn prediction package.

Modules:
    * :mod:`churn.config` – paths, schema and MLflow settings
    * :mod:`churn.dataset` – data loading, cleaning and splitting
    * :mod:`churn.features` – feature engineering and preprocessing pipeline
    * :mod:`churn.train` – model training with MLflow tracking
    * :mod:`churn.registry` – MLflow Model Registry helpers
    * :mod:`churn.predict` – batch inference
    * :mod:`churn.explain` – SHAP explainability
"""

__version__ = "1.0.0"
