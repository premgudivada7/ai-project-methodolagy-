"""Run a small grid of experiments to compare models in the MLflow UI.

Usage::

    python -m churn.experiments
"""

import logging

import pandas as pd

from churn import config
from churn.train import parse_args, train

logger = logging.getLogger(__name__)

EXPERIMENTS = [
    ["--model", "logreg", "--C", "1.0", "--run-name", "logreg_C1"],
    ["--model", "logreg", "--C", "0.1", "--run-name", "logreg_C0.1"],
    ["--model", "rf", "--n-estimators", "100", "--max-depth", "6", "--run-name", "rf_100_d6"],
    ["--model", "rf", "--n-estimators", "300", "--max-depth", "12", "--run-name", "rf_300_d12"],
    ["--model", "rf", "--n-estimators", "500", "--max-depth", "0", "--run-name", "rf_500_full"],
    [
        "--model",
        "gb",
        "--n-estimators",
        "200",
        "--max-depth",
        "3",
        "--learning-rate",
        "0.1",
        "--class-weight",
        "none",
        "--run-name",
        "gb_200_d3",
    ],
    [
        "--model",
        "gb",
        "--n-estimators",
        "400",
        "--max-depth",
        "5",
        "--learning-rate",
        "0.05",
        "--class-weight",
        "none",
        "--run-name",
        "gb_400_d5",
    ],
]


def main() -> None:
    """Train every configuration and print a comparison table."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    results = [train(parse_args(argv)) for argv in EXPERIMENTS]
    table = pd.DataFrame(results).set_index("run_name")
    cols = ["cv_pr_auc", "roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy", "lift_top10"]
    table = table[cols].round(3).sort_values("cv_pr_auc", ascending=False)
    table.to_csv(config.REPORTS_DIR / "experiments_comparison.csv")
    print(table.to_string())


if __name__ == "__main__":
    main()
