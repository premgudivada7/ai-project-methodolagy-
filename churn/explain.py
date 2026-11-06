"""Explainable AI step (Part 3): SHAP explanations of the churn model.

Builds a ``shap.TreeExplainer`` on the tree model of the pipeline, computes
Shapley values on the test set and saves every requested visualisation in
``reports/figures/shap``:

* force plot for one customer (local explanation),
* stacked force plot for all customers at once (+ heatmap),
* summary plot for each class (beeswarm per class + multi-class bar),
* for one customer: waterfall plot, force plot, mean |SHAP| bar plot,
  beeswarm plot and dependence plots (with the customer highlighted).

Usage::

    python -m churn.explain                         # champion model from MLflow
    python -m churn.explain --model-path models/rf_300_d12.joblib --index 12
"""

import argparse
import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from churn import config  # noqa: E402
from churn.features import split_xy  # noqa: E402
from churn.predict import load_model  # noqa: E402

logger = logging.getLogger(__name__)
CLASS_NAMES = ["Stays (0)", "Churns (1)"]


def save(name: str, fig=None) -> None:
    """Save the current matplotlib figure into the SHAP figures folder."""
    fig = fig or plt.gcf()
    fig.savefig(config.SHAP_DIR / name, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("saved %s", name)


def transform(pipe, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the pipeline preprocessing and keep readable feature names."""
    pre = pipe.named_steps["preprocess"]
    data = pre.transform(X)
    return pd.DataFrame(data, columns=pre.get_feature_names_out(), index=X.index)


def per_class_explanations(explanation) -> list:
    """Return one :class:`shap.Explanation` per class.

    Random forests return values of shape ``(n, features, 2)``; gradient
    boosting returns ``(n, features)`` for the positive class only (log-odds),
    in which case class 0 is the opposite of class 1.
    """
    if explanation.values.ndim == 3:
        return [explanation[:, :, k] for k in range(explanation.values.shape[2])]
    neg = shap.Explanation(
        values=-explanation.values,
        base_values=-np.asarray(explanation.base_values),
        data=explanation.data,
        feature_names=explanation.feature_names,
    )
    return [neg, explanation]


def pick_customer(proba: np.ndarray, y: pd.Series, index=None) -> int:
    """Choose the customer to explain.

    By default: the true churner with the highest predicted risk.
    """
    if index is not None:
        return int(index)
    churners = np.where(y.values == 1)[0]
    return int(churners[np.argmax(proba[churners])])


def reasons_text(exp_row, top: int = 3) -> list:
    """Plain-language top reasons pushing a customer towards churn."""
    order = np.argsort(-exp_row.values)[:top]
    return [
        f"{exp_row.feature_names[i]} = {exp_row.data[i]:.2f} (+{exp_row.values[i]:.3f})"
        for i in order
        if exp_row.values[i] > 0
    ]


def run(model_uri=None, model_path=None, index=None, max_rows=None, log_mlflow=True) -> dict:
    """Compute SHAP values and generate all plots.

    Args:
        model_uri: MLflow model URI (default: registry champion).
        model_path: local joblib model (takes precedence).
        index: position (in the test set) of the customer to explain.
        max_rows: optionally limit the number of test rows explained.
        log_mlflow: log the figures as artifacts of a new MLflow run.

    Returns:
        Summary dict (chosen customer, probability, top reasons).
    """
    config.SHAP_DIR.mkdir(parents=True, exist_ok=True)
    pipe = load_model(model_uri, model_path)
    test = pd.read_csv(config.TEST_FILE)
    if max_rows:
        test = test.sample(min(max_rows, len(test)), random_state=config.RANDOM_STATE)
    X, y = split_xy(test)
    X_t = transform(pipe, X)
    clf = pipe.named_steps["model"]
    proba = pipe.predict_proba(X)[:, 1]

    # ---- 1. TreeExplainer and Shapley values ---------------------------
    explainer = shap.TreeExplainer(clf)
    explanation = explainer(X_t)
    exp0, exp1 = per_class_explanations(explanation)
    pd.DataFrame(exp1.values, columns=X_t.columns).assign(
        **{config.ID_COL: test[config.ID_COL].values, "churn_probability": proba}
    ).to_csv(config.REPORTS_DIR / "shap_values_class1.csv", index=False)
    X_t.to_csv(config.REPORTS_DIR / "shap_feature_values.csv", index=False)

    i = pick_customer(proba, y, index)
    cust = test.iloc[i]
    logger.info(
        "Explaining customer %s (p=%.3f, churn=%s)", cust[config.ID_COL], proba[i], y.iloc[i]
    )
    base1 = float(np.ravel(exp1.base_values)[0])

    # ---- 2. Force plot for one point -------------------------------------
    shap.force_plot(
        base1,
        exp1.values[i],
        X_t.iloc[i].round(2),
        matplotlib=True,
        show=False,
        figsize=(20, 3.5),
        text_rotation=15,
    )
    save("01_force_plot_single_customer.png")
    shap.save_html(
        str(config.SHAP_DIR / "01_force_plot_single_customer.html"),
        shap.force_plot(base1, exp1.values[i], X_t.iloc[i].round(2)),
    )

    # ---- 3. All points at once ----------------------------------------
    shap.save_html(
        str(config.SHAP_DIR / "02_force_plot_all_customers.html"),
        shap.force_plot(base1, exp1.values, X_t.round(2)),
    )
    shap.plots.heatmap(exp1, max_display=12, show=False)
    save("02_heatmap_all_customers.png")

    # ---- 4. Summary plot for each class -------------------------------
    for k, (exp_k, name) in enumerate(zip((exp0, exp1), CLASS_NAMES)):
        shap.summary_plot(exp_k.values, X_t, max_display=15, show=False)
        plt.title(f"SHAP summary - class {name}")
        save(f"03_summary_class{k}.png")
    # Mean |SHAP| per class as a grouped bar chart (drawn with matplotlib so it
    # does not depend on how each SHAP version handles multi-class input)
    mean_abs = pd.DataFrame(
        {name: np.abs(e.values).mean(0) for e, name in zip((exp0, exp1), CLASS_NAMES)},
        index=X_t.columns,
    )
    mean_abs = mean_abs.loc[mean_abs.sum(1).sort_values().index[-15:]]
    ax = mean_abs.plot.barh(figsize=(8, 7), color=["#2E75B6", "#C0504D"])
    ax.set_xlabel("mean |SHAP value|")
    plt.title("Mean |SHAP| per class")
    plt.tight_layout()
    save("03_summary_bar_both_classes.png")

    # ---- 5. Specific point: waterfall ----------------------------------
    shap.plots.waterfall(exp1[i], max_display=12, show=False)
    plt.title(f"Waterfall - customer {int(cust[config.ID_COL])} (p={proba[i]:.2f})")
    save("04_waterfall_customer.png")

    # ---- 6. Mean SHAP plot (global) and SHAP bar for the customer -------
    shap.plots.bar(exp1, max_display=15, show=False)
    plt.title("Mean |SHAP value| (class churn)")
    save("05_mean_shap_bar.png")
    shap.plots.bar(exp1[i], max_display=12, show=False)
    plt.title(f"SHAP values - customer {int(cust[config.ID_COL])}")
    save("05_shap_bar_customer.png")

    # ---- 7. Beeswarm ---------------------------------------------------
    shap.plots.beeswarm(exp1, max_display=15, show=False)
    plt.title("Beeswarm (class churn)")
    save("06_beeswarm.png")

    # ---- 8. Dependence plots, customer highlighted -----------------------
    top_features = list(pd.Series(np.abs(exp1.values).mean(0), index=X_t.columns).nlargest(4).index)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, feat in zip(axes.ravel(), top_features):
        shap.dependence_plot(feat, exp1.values, X_t, interaction_index="auto", ax=ax, show=False)
        j = X_t.columns.get_loc(feat)
        ax.scatter(
            X_t.iloc[i, j],
            exp1.values[i, j],
            s=160,
            facecolors="none",
            edgecolors="red",
            linewidths=2.5,
            label="selected customer",
            zorder=5,
        )
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Dependence plots - top 4 features (red circle = selected customer)")
    fig.tight_layout()
    save("07_dependence_plots.png", fig)
    for feat in top_features:
        shap.dependence_plot(feat, exp1.values, X_t, show=False)
        save(f"07_dependence_{feat}.png")

    summary = {
        "customer_id": int(cust[config.ID_COL]),
        "test_position": i,
        "churn_probability": float(proba[i]),
        "actual_churn": int(y.iloc[i]),
        "base_value_class1": base1,
        "top_reasons": reasons_text(exp1[i]),
        "global_top_features": top_features,
    }
    with open(config.SHAP_DIR / "explanation_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    if log_mlflow:
        import mlflow

        from churn.train import setup_mlflow

        setup_mlflow()
        with mlflow.start_run(run_name="shap-explanations"):
            mlflow.set_tag("step", "xai")
            mlflow.log_params({"model": model_path or model_uri or "champion", "rows": len(X_t)})
            mlflow.log_artifacts(str(config.SHAP_DIR), artifact_path="shap")
    logger.info("Top reasons: %s", summary["top_reasons"])
    return summary


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-uri", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--index", type=int, default=None, help="test-set position of the customer")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run(args.model_uri, args.model_path, args.index, args.max_rows, not args.no_mlflow)


if __name__ == "__main__":
    main()
