"""MLflow Model Registry helpers.

Selects the best run of the experiment, registers its model as a new
version of ``retailgenius-churn-model`` and gives it the ``champion`` alias
(the version used for serving).

Usage::

    python -m churn.registry --metric cv_pr_auc_mean
    python -m churn.registry --list
"""

import argparse
import logging

import mlflow
from mlflow import MlflowClient

from churn import config
from churn.train import setup_mlflow

logger = logging.getLogger(__name__)


def best_run(metric: str = "cv_pr_auc_mean"):
    """Return the run with the highest value of ``metric``.

    Args:
        metric: metric to maximise (as logged by :mod:`churn.train`).

    Returns:
        The best :class:`mlflow.entities.Run`.
    """
    exp = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    runs = MlflowClient().search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No finished run found: train a model first.")
    return runs[0]


def register_best(metric: str = "cv_pr_auc_mean"):
    """Register the best run's model and set the ``champion`` alias.

    Returns:
        The created :class:`~mlflow.entities.model_registry.ModelVersion`.
    """
    client = MlflowClient()
    run = best_run(metric)
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    # MLflow 3 stores models as "logged models"; fall back on them if needed
    try:
        logged = mlflow.search_logged_models(
            filter_string=f"source_run_id = '{run_id}'", output_format="list"
        )
        if logged:
            model_uri = logged[0].model_uri
    except Exception:  # noqa: BLE001 - MLflow 2.x has no logged models API
        pass
    version = mlflow.register_model(model_uri, config.REGISTERED_MODEL_NAME)
    client.set_registered_model_alias(
        config.REGISTERED_MODEL_NAME, config.CHAMPION_ALIAS, version.version
    )
    client.update_model_version(
        config.REGISTERED_MODEL_NAME,
        version.version,
        description=(
            f"Run '{run.info.run_name}' - "
            f"{metric}={run.data.metrics.get(metric, float('nan')):.3f}, "
            f"recall={run.data.metrics.get('test_recall', float('nan')):.3f}"
        ),
    )
    client.set_model_version_tag(
        config.REGISTERED_MODEL_NAME, version.version, "validation_status", "approved"
    )
    logger.info(
        "Registered %s version %s from run %s (%s) with alias '%s'",
        config.REGISTERED_MODEL_NAME,
        version.version,
        run_id,
        run.info.run_name,
        config.CHAMPION_ALIAS,
    )
    return version


def list_versions() -> None:
    """Print all registered versions with their aliases."""
    client = MlflowClient()
    model = client.get_registered_model(config.REGISTERED_MODEL_NAME)
    aliases = {}
    for alias, ver in (model.aliases or {}).items():
        aliases.setdefault(str(ver), []).append(alias)
    for mv in client.search_model_versions(f"name='{config.REGISTERED_MODEL_NAME}'"):
        print(
            f"v{mv.version:>3} | run {mv.run_id} | aliases {aliases.get(str(mv.version), [])} "
            f"| {mv.description or ''}"
        )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", default="cv_pr_auc_mean")
    parser.add_argument("--list", action="store_true", help="only list versions")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    setup_mlflow()
    if not args.list:
        register_best(args.metric)
    list_versions()


if __name__ == "__main__":
    main()
