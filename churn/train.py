"""Model training step with MLflow tracking.

Each call trains one model, evaluates it on the hold-out test set and logs
parameters, metrics, plots and the model itself to MLflow.

Usage::

    python -m churn.train --model rf --n-estimators 300 --max-depth 12
    python -m churn.train --model logreg --C 0.5
    python -m churn.train --model gb --learning-rate 0.1 --register
"""

import argparse
import hashlib
import inspect
import json
import logging

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from churn import config  # noqa: E402
from churn.features import build_preprocessor, split_xy  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_CHOICES = ("logreg", "rf", "gb", "xgb")
# Types the skops serializer (MLflow >= 3.x) must accept to save our pipeline
SKOPS_TRUSTED_TYPES = ["numpy.dtype"]


def build_model(args) -> Pipeline:
    """Create the full (preprocessing + classifier) pipeline.

    Args:
        args: parsed command-line arguments with the hyper-parameters.

    Returns:
        An unfitted scikit-learn :class:`~sklearn.pipeline.Pipeline`.
    """
    class_weight = None if args.class_weight == "none" else args.class_weight
    if args.model == "logreg":
        clf = LogisticRegression(C=args.C, class_weight=class_weight, max_iter=2000)
    elif args.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        )
    elif args.model == "gb":
        clf = GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth or 3,
            learning_rate=args.learning_rate,
            min_samples_leaf=args.min_samples_leaf,
            random_state=config.RANDOM_STATE,
        )
    elif args.model == "xgb":
        from xgboost import XGBClassifier  # optional dependency

        clf = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth or 6,
            learning_rate=args.learning_rate,
            eval_metric="logloss",
            random_state=config.RANDOM_STATE,
        )
    else:  # pragma: no cover - argparse already restricts choices
        raise ValueError(args.model)
    return Pipeline(
        [
            ("preprocess", build_preprocessor(scale=args.model == "logreg")),
            ("model", clf),
        ]
    )


def evaluate(y_true, proba, threshold: float = 0.5) -> dict:
    """Compute classification metrics.

    Args:
        y_true: true labels (0 / 1).
        proba: predicted churn probabilities.
        threshold: decision threshold used to turn probabilities into labels.

    Returns:
        dict of metric name -> value.
    """
    pred = (proba >= threshold).astype(int)
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    best = int(np.argmax(f1s))
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "lift_top10": lift_at(y_true, proba, 0.10),
        "best_f1_threshold": float(thr[best]),
        "best_f1": float(f1s[best]),
    }


def lift_at(y_true, proba, share: float) -> float:
    """Lift of the top ``share`` highest-risk customers vs. random targeting.

    Args:
        y_true: true labels.
        proba: predicted probabilities.
        share: share of the population targeted (e.g. 0.1 for top 10%).

    Returns:
        churn rate in the top group divided by the overall churn rate.
    """
    y_true = np.asarray(y_true)
    n_top = max(1, int(len(y_true) * share))
    top = np.argsort(-np.asarray(proba))[:n_top]
    return float(y_true[top].mean() / max(y_true.mean(), 1e-12))


def make_plots(y_true, proba, threshold, pipe) -> dict:
    """Create evaluation figures.

    Returns:
        dict of artifact file name -> matplotlib figure.
    """
    figs = {}
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, (proba >= threshold).astype(int), ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Confusion matrix (test)")
    figs["confusion_matrix.png"] = fig

    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax)
    ax.set_title("ROC curve (test)")
    figs["roc_curve.png"] = fig

    fig, ax = plt.subplots(figsize=(4.5, 4))
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=ax)
    ax.set_title("Precision-recall curve (test)")
    figs["pr_curve.png"] = fig

    clf = pipe.named_steps["model"]
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=names)
    elif hasattr(clf, "coef_"):
        imp = pd.Series(np.abs(clf.coef_[0]), index=names)
    else:
        imp = None
    if imp is not None:
        imp = imp.sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(6, 5))
        imp.plot.barh(ax=ax, color="#2E75B6")
        ax.set_title("Top 15 feature importances")
        fig.tight_layout()
        figs["feature_importance.png"] = fig
    return figs


def file_md5(path) -> str:
    """MD5 hash of a file, logged to trace the exact training data version."""
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def setup_mlflow() -> None:
    """Point MLflow to the project tracking store and experiment."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(config.EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            config.EXPERIMENT_NAME, artifact_location=config.MLFLOW_ARTIFACT_LOCATION
        )
    mlflow.set_experiment(config.EXPERIMENT_NAME)


def log_model(pipe, X_train, register: bool):
    """Log the fitted pipeline as an MLflow model.

    The pyfunc flavour returns ``predict_proba`` so the served model outputs
    churn probabilities. Works with MLflow 2.x and 3.x:

    * MLflow >= 3 uses ``name=`` instead of ``artifact_path=``;
    * recent MLflow versions save scikit-learn models with *skops*, which
      refuses types it does not know (here ``numpy.dtype``, used by the
      one-hot encoder). We declare it as trusted, and fall back to
      cloudpickle if skops still refuses the model.
    """
    params = inspect.signature(mlflow.sklearn.log_model).parameters
    signature = infer_signature(X_train, pipe.predict_proba(X_train.head(50)))
    kwargs = dict(
        sk_model=pipe,
        signature=signature,
        input_example=X_train.head(3),
        pyfunc_predict_fn="predict_proba",
        registered_model_name=config.REGISTERED_MODEL_NAME if register else None,
    )
    kwargs["name" if "name" in params else "artifact_path"] = "model"
    if "skops_trusted_types" in params:
        kwargs["skops_trusted_types"] = SKOPS_TRUSTED_TYPES
    try:
        return mlflow.sklearn.log_model(**kwargs)
    except MlflowException as exc:
        if "untrusted" not in str(exc).lower() or "serialization_format" not in params:
            raise
        logger.warning("skops refused the model (%s); saving it with cloudpickle", exc)
        kwargs.pop("skops_trusted_types", None)
        kwargs["serialization_format"] = "cloudpickle"
        return mlflow.sklearn.log_model(**kwargs)


def train(args) -> dict:
    """Train, evaluate and log one model run.

    Args:
        args: parsed command-line arguments.

    Returns:
        dict with the run id and test metrics.
    """
    train_df = pd.read_csv(config.TRAIN_FILE)
    test_df = pd.read_csv(config.TEST_FILE)
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    setup_mlflow()
    run_name = args.run_name or f"{args.model}"
    with mlflow.start_run(run_name=run_name) as run:
        params = {k: v for k, v in vars(args).items() if k not in {"register", "run_name"}}
        mlflow.log_params(params)
        mlflow.set_tags(
            {
                "model_type": args.model,
                "train_data_md5": file_md5(config.TRAIN_FILE),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "project": "RetailGenius churn",
            }
        )
        try:
            mlflow.log_input(mlflow.data.from_pandas(train_df, name="train"), context="training")
        except Exception:  # noqa: BLE001 - dataset logging is optional
            pass

        pipe = build_model(args)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
        # Model selection uses cross-validation on the training set only, so
        # the test set stays untouched until the final evaluation.
        scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
        cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        for name in ("roc_auc", "pr_auc"):
            mlflow.log_metric(f"cv_{name}_mean", float(cv_res[f"test_{name}"].mean()))
            mlflow.log_metric(f"cv_{name}_std", float(cv_res[f"test_{name}"].std()))

        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, proba, args.threshold)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        for fname, fig in make_plots(y_test, proba, args.threshold, pipe).items():
            mlflow.log_figure(fig, f"plots/{fname}")
            fig.savefig(config.FIGURES_DIR / f"{run_name}_{fname}", dpi=120, bbox_inches="tight")
            plt.close(fig)

        log_model(pipe, X_train, args.register)

        config.MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(pipe, config.MODELS_DIR / f"{run_name}.joblib")
        result = {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "cv_pr_auc": float(cv_res["test_pr_auc"].mean()),
            **metrics,
        }
        (config.REPORTS_DIR / "metrics").mkdir(exist_ok=True)
        with open(config.REPORTS_DIR / "metrics" / f"{run_name}.json", "w") as fh:
            json.dump(result, fh, indent=2)
        logger.info(
            "%s | roc_auc=%.3f pr_auc=%.3f recall=%.3f precision=%.3f f1=%.3f",
            run_name,
            metrics["roc_auc"],
            metrics["pr_auc"],
            metrics["recall"],
            metrics["precision"],
            metrics["f1"],
        )
    return result


def parse_args(argv=None):
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", choices=MODEL_CHOICES, default="rf")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=None)
    p.add_argument("--min-samples-leaf", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--class-weight", choices=("none", "balanced"), default="balanced")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--run-name", default=None)
    p.add_argument(
        "--register", action="store_true", help="register model in the MLflow Model Registry"
    )
    args = p.parse_args(argv)
    if args.max_depth is not None and args.max_depth <= 0:
        args.max_depth = None
    return args


def main(argv=None) -> None:
    """Command-line entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    train(parse_args(argv))


if __name__ == "__main__":
    main()
