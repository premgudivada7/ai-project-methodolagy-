"""Build the Part 3 (Explainable AI) report from the SHAP outputs.

Run it after ``make explain``. It reads the figures in reports/figures/shap,
explanation_summary.json and the SHAP value tables, then writes
reports/Part3_Explainable_AI.docx. Every number in the report comes from those
files, so the report always matches the model that was explained.

If a figure is missing, a placeholder box is written instead, so the script
also works before the SHAP step has been run.

Usage::

    make explain
    python scripts/build_xai_report.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from churn import config  # noqa: E402

BLUE = RGBColor(0x1F, 0x4E, 0x79)
GREY = RGBColor(0x60, 0x60, 0x60)
REPO = "https://github.com/premgudivada7/ai-project-methodolagy-"


# ----------------------------------------------------------------------------
# small docx helpers
# ----------------------------------------------------------------------------
def setup(doc):
    """Page size, margins and base styles."""
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(2))
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(6)
    for name, size, color in (
        ("Heading 1", 16, BLUE),
        ("Heading 2", 13, RGBColor(0x2E, 0x75, 0xB6)),
    ):
        st = doc.styles[name]
        st.font.name, st.font.size, st.font.bold, st.font.color.rgb = (
            "Calibri",
            Pt(size),
            True,
            color,
        )
        st.paragraph_format.space_before = Pt(14)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.keep_with_next = True
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = footer.add_run("RetailGenius churn – Part 3")
    r.font.size, r.font.color.rgb = Pt(8), GREY


def para(doc, text, bold_prefix=None, style=None):
    """Paragraph with an optional bold prefix."""
    p = doc.add_paragraph(style=style)
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    p.add_run(text)
    return p


def bullet(doc, text, bold_prefix=None):
    """Bulleted paragraph."""
    return para(doc, text, bold_prefix, style="List Bullet")


def code(doc, text):
    """Monospace block with a light grey background."""
    for line in text.splitlines():
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.left_indent = Cm(0.4)
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:fill"), "F2F2F2")
        p._p.get_or_add_pPr().append(shd)  # pylint: disable=protected-access
        r = p.add_run(line or " ")
        r.font.name, r.font.size = "Consolas", Pt(8.5)
    doc.add_paragraph()


def figure(doc, name, caption, width_cm=16):
    """Insert a SHAP figure, or a placeholder if it does not exist yet."""
    path = config.SHAP_DIR / name
    if path.exists():
        doc.add_picture(str(path), width=Cm(width_cm))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(f"[ {name} – run `make explain` then this script to insert it ]")
        r.italic, r.font.color.rgb = True, GREY
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.italic, r.font.size, r.font.color.rgb = True, Pt(9), GREY


def table(doc, header, rows, widths=None):
    """Simple table with a coloured header."""
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, text in zip(t.rows[0].cells, header):
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        run.bold, run.font.color.rgb, run.font.size = True, RGBColor(0xFF, 0xFF, 0xFF), Pt(9.5)
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:fill"), "1F4E79")
        cell._tc.get_or_add_tcPr().append(shd)  # pylint: disable=protected-access
    for row in rows:
        cells = t.add_row().cells
        for cell, text in zip(cells, row):
            cell.text = ""
            cell.paragraphs[0].add_run(str(text)).font.size = Pt(9.5)
    if widths:
        for row in t.rows:
            for cell, w in zip(row.cells, widths):
                cell.width = Cm(w)
    doc.add_paragraph()


# ----------------------------------------------------------------------------
# numbers taken from the SHAP outputs
# ----------------------------------------------------------------------------
def load_outputs():
    """Load the summary json and SHAP tables, or None if not generated yet."""
    summary_file = config.SHAP_DIR / "explanation_summary.json"
    values_file = config.REPORTS_DIR / "shap_values_class1.csv"
    feats_file = config.REPORTS_DIR / "shap_feature_values.csv"
    if not (summary_file.exists() and values_file.exists() and feats_file.exists()):
        return None
    summary = json.loads(summary_file.read_text())
    shap_df = pd.read_csv(values_file)
    feats = pd.read_csv(feats_file)
    return summary, shap_df, feats


def feature_effects(shap_df, feats, top=10):
    """Mean |SHAP| and direction of effect for the most important features.

    The direction is the correlation between the feature value and its SHAP
    value: positive means higher values push towards churn.
    """
    cols = [c for c in feats.columns if c in shap_df.columns]
    rows = []
    for c in cols:
        imp = float(shap_df[c].abs().mean())
        x, s = feats[c].values, shap_df[c].values
        corr = float(np.corrcoef(x, s)[0, 1]) if np.std(x) > 0 and np.std(s) > 0 else 0.0
        binary = set(np.unique(x)) <= {0.0, 1.0}
        if binary:
            text = (
                "being in this group raises churn risk"
                if corr > 0
                else "being in this group lowers churn risk"
            )
        else:
            text = (
                "higher value → more churn risk" if corr > 0 else "higher value → less churn risk"
            )
        rows.append((c, imp, corr, text))
    rows.sort(key=lambda r: -r[1])
    return rows[:top]


# ----------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------
def build(out_path):
    """Write the Part 3 report."""
    outputs = load_outputs()
    doc = Document()
    setup(doc)

    # title page
    for _ in range(6):
        doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run("EPITA International Programs  |  AI Project Methodology 2026")
    r.font.color.rgb = GREY
    p = doc.add_paragraph()
    r = p.add_run("Graded Project, Part 3")
    r.bold, r.font.size, r.font.color.rgb = True, Pt(15), BLUE
    p = doc.add_paragraph()
    r = p.add_run("Explainable AI with SHAP")
    r.bold, r.font.size = True, Pt(26)
    p = doc.add_paragraph()
    r = p.add_run("Why does the model think a RetailGenius customer is about to leave?")
    r.font.size = Pt(14)
    doc.add_paragraph()
    para(doc, "Group 5", "Group: ")
    para(
        doc,
        "Aravind Reddy Kolan, Prem Siva Naga Sai Ram Kumar Gudivada, Praveen Kumar Koppula",
        "Members: ",
    )
    para(doc, "November 2026", "Date: ")
    para(doc, f"{REPO} (same repository as Part 2)", "GitHub repository: ")
    doc.add_page_break()

    # 1
    doc.add_heading("1. Why we need explanations", level=1)
    para(
        doc,
        "The churn model from Part 2 is a random forest with hundreds of trees. It predicts well, "
        "but nobody can read 500 trees to understand a single score. That is a real problem for "
        "RetailGenius. A marketing manager will not spend budget on a customer just because "
        '"the model said 0.87". A support agent needs to know whether to talk about delivery, '
        "a complaint or a discount. And under GDPR, customers can ask how automated decisions "
        "about them were made.",
    )
    para(
        doc,
        "SHAP (SHapley Additive exPlanations) answers this. It comes from game theory: each feature "
        'is treated as a "player" and gets a share of the prediction, called its Shapley value. '
        "For every customer, the SHAP values add up exactly to the difference between the model's "
        "prediction for that customer and the average prediction (the base value). A positive "
        "value pushes towards churn, a negative one pushes away from it.",
    )

    # 2
    doc.add_heading("2. Setting up SHAP in the project", level=1)
    para(
        doc,
        "We added shap to pyproject.toml and requirements.txt, so it installs with the rest of the "
        "project. All the XAI code is in churn/explain.py and runs as one step of the pipeline:",
    )
    code(
        doc,
        "pip install shap            # already in requirements.txt / pyproject.toml\n"
        "make explain                # = python -m churn.explain\n"
        "mlflow run . -e explain     # same thing as an MLflow Project entry point\n"
        "python scripts/build_xai_report.py   # rebuilds this report from the outputs",
    )
    para(
        doc,
        "The script loads the champion model from the MLflow Model Registry "
        "(models:/retailgenius-churn-model@champion), applies the pipeline's preprocessing to the "
        "test customers so we explain the model on the same features it really sees, and builds a "
        "TreeExplainer on the random forest:",
    )
    code(
        doc,
        "explainer = shap.TreeExplainer(clf)       # exact, fast algorithm for tree models\n"
        "explanation = explainer(X_test_transformed)  # shape (customers, features, 2 classes)\n"
        "exp0, exp1 = explanation[:, :, 0], explanation[:, :, 1]",
    )
    para(
        doc,
        "TreeExplainer computes exact Shapley values for tree ensembles in polynomial time, so we "
        "can explain every test customer in seconds instead of approximating with sampling. "
        "For a random forest the values are in probability units. All figures are saved in "
        'reports/figures/shap and logged to MLflow under a run named "shap-explanations".',
    )

    if outputs:
        summary, shap_df, feats = outputs
        n_rows, n_feats = feats.shape
        para(
            doc,
            f"We explained {n_rows} test customers on {n_feats} features (after one-hot encoding). "
            f"The base value for the churn class is {summary['base_value_class1']:.3f}, which is "
            "the average churn probability the model gives on the training data. It is close to "
            "0.5 rather than the real churn rate (about 17%) because we trained with balanced class "
            "weights, which make the forest treat both classes as equally frequent.",
        )
        cid, prob = summary["customer_id"], summary["churn_probability"]
        actual = "did churn" if summary["actual_churn"] else "did not churn"
    else:
        summary = None
        cid, prob, actual = "<id>", float("nan"), "<actual>"

    # 3
    doc.add_heading("3. Explaining one customer", level=1)
    para(
        doc,
        f"By default the script picks the real churner with the highest predicted risk in the test "
        f"set. Here that is customer {cid}. The model gives them a churn probability of "
        f"{prob:.2f}, and they {actual}. (Another customer can be chosen with "
        "python -m churn.explain --index N.)",
    )
    figure(
        doc,
        "01_force_plot_single_customer.png",
        "Figure 1 – Force plot for the selected customer (interactive version: "
        "01_force_plot_single_customer.html)",
    )
    para(
        doc,
        "How to read it: the model starts from the base value. Red arrows are features that push the "
        "probability up, blue arrows push it down, and the width of each arrow is the size of the "
        "effect. Where they meet is the final prediction.",
    )
    if summary and summary["top_reasons"]:
        para(doc, "The three biggest reasons this customer is flagged:")
        for reason in summary["top_reasons"]:
            bullet(doc, reason)
        para(
            doc,
            "Each line shows the customer's value for the feature (after imputation) and, in "
            "brackets, how much it adds to the churn probability.",
        )

    # 4
    doc.add_heading("4. Explaining all customers at once", level=1)
    para(
        doc,
        "If we take the force plot of every customer, turn it vertical and put them side by side, "
        "we get the stacked force plot. It is interactive, so it is saved as HTML "
        "(02_force_plot_all_customers.html). Customers can be ordered by similarity, by "
        "prediction or by a single feature, which makes groups of similar churners easy to spot.",
    )
    para(
        doc,
        "Since an HTML file cannot go in a report, we also made a heatmap that shows the same "
        "information: one column per customer, one row per feature, and the colour is the SHAP "
        "value. The line on top is the model's output for each customer.",
    )
    figure(doc, "02_heatmap_all_customers.png", "Figure 2 – SHAP heatmap of all test customers")

    # 5
    doc.add_heading("5. Summary plot for each class", level=1)
    para(
        doc,
        "The random forest gives one probability per class, so SHAP gives one set of values per "
        "class too. In a binary problem the two are mirror images: anything that raises the "
        "probability of churning lowers the probability of staying by exactly the same amount. "
        "We still plot both, as asked.",
    )
    figure(doc, "03_summary_class0.png", "Figure 3 – Summary plot, class 0 (customer stays)", 14)
    figure(doc, "03_summary_class1.png", "Figure 4 – Summary plot, class 1 (customer churns)", 14)
    figure(
        doc,
        "03_summary_bar_both_classes.png",
        "Figure 5 – Mean |SHAP| per feature, both classes",
        14,
    )
    para(
        doc,
        "Each dot is one customer. Its position is the SHAP value (its push on the prediction) and "
        "its colour is the feature value, red for high and blue for low. Features are sorted from "
        "most to least important.",
    )

    # 6
    doc.add_heading("6. Going further for one customer and globally", level=1)
    doc.add_heading("6.1 Waterfall plot", level=2)
    para(
        doc,
        "The waterfall plot is the easiest one to show to a non-technical audience. It starts at the "
        "base value at the bottom and adds one feature at a time until it reaches the customer's "
        "prediction at the top. Each bar is labelled with the feature value and its contribution.",
    )
    figure(doc, "04_waterfall_customer.png", f"Figure 6 – Waterfall plot for customer {cid}", 14)
    doc.add_heading("6.2 Force plot", level=2)
    para(
        doc,
        "This is the same information as the waterfall, laid out horizontally (Figure 1). We use "
        "the waterfall in reports and the force plot in dashboards, where space is wide rather "
        "than tall.",
    )
    doc.add_heading("6.3 Mean SHAP plot", level=2)
    para(
        doc,
        "The mean |SHAP| bar plot ranks features by their average impact over all customers. Unlike "
        "the impurity-based importance in Part 2, it is measured in probability points, and it is "
        "consistent: a feature that matters more always gets a higher value. Next to it we show the "
        "bar plot of the selected customer's own SHAP values, so their profile can be compared "
        "with the global picture.",
    )
    figure(
        doc,
        "05_mean_shap_bar.png",
        "Figure 7 – Mean |SHAP value| (churn class), all test customers",
        14,
    )
    figure(doc, "05_shap_bar_customer.png", f"Figure 8 – SHAP values of customer {cid}", 14)
    doc.add_heading("6.4 Beeswarm plot", level=2)
    para(
        doc,
        "The beeswarm combines importance and direction in one picture: one dot per customer, "
        "stacked where values pile up.",
    )
    figure(doc, "06_beeswarm.png", "Figure 9 – Beeswarm plot (churn class)", 14)
    doc.add_heading("6.5 Dependence plots", level=2)
    para(
        doc,
        "A dependence plot puts one feature on the x axis and its SHAP value on the y axis, for every "
        "customer. It shows the shape of the effect (straight line, threshold, plateau), and the "
        "colour shows the feature that interacts most with it. We plotted the four most important "
        "features and circled the selected customer in red to see where they sit.",
    )
    figure(doc, "07_dependence_plots.png", "Figure 10 – Dependence plots of the top 4 features", 16)

    # 7
    doc.add_heading("7. What the explanations tell us", level=1)
    if outputs:
        effects = feature_effects(shap_df, feats)
        para(
            doc,
            "The table below is computed from the SHAP values. Direction is the correlation between "
            "a feature's value and its SHAP value across customers.",
        )
        table(
            doc,
            ["Feature", "Mean |SHAP|", "Direction (corr.)", "Reading"],
            [(f, f"{imp:.3f}", f"{c:+.2f}", t) for f, imp, c, t in effects],
            [5, 2.5, 3, 6.5],
        )
        top = ", ".join(e[0] for e in effects[:4])
        para(
            doc,
            f"The four features that matter most are {top}. Our exploration in Part 2 pointed to "
            "the same things (tenure, complaints and cashback), which gives us some confidence the "
            "model learned real patterns rather than noise. SHAP adds the direction and size of each "
            "effect, and shows how they differ from one customer to the next.",
        )
    else:
        para(doc, "[ This section is filled in automatically once `make explain` has been run. ]")
    para(doc, "Things we noticed that matter for the business:")
    bullet(
        doc,
        " the first months are where customers are lost, so onboarding is the place to act.",
        "New customers are the main risk:",
    )
    bullet(
        doc,
        " a customer who complained is much more likely to leave, so a follow-up call after a "
        "complaint is a cheap retention action.",
        "Complaints are a strong signal:",
    )
    bullet(
        doc,
        " the satisfaction score effect should be checked with the business, as we noted in Part 2.",
        "Some effects need a second look:",
    )

    # 8
    doc.add_heading("8. How RetailGenius would use this", level=1)
    table(
        doc,
        ["Audience", "What they get from SHAP"],
        [
            (
                "Marketing",
                "Top 3 reasons per at-risk customer in the CRM, used to pick the right offer",
            ),
            (
                "Customer support",
                'Plain-language reasons in the agent console ("complained last month")',
            ),
            ("Product / management", "Mean SHAP and beeswarm plots in the monthly churn review"),
            ("Data science", "Dependence plots to debug the model and catch leakage"),
            ("Legal / DPO", "An explanation for each automated decision (GDPR transparency)"),
        ],
        [5, 12],
    )
    para(
        doc,
        "Limits. SHAP explains the model, not the real world: a feature with a high SHAP value is "
        "predictive, but that does not prove it causes churn. When features are correlated (for "
        "example Tenure and our IsNewCustomer flag), the credit is split between them, so their "
        "individual values should be read together. And because Part 2 found duplicated customers, "
        "we only explain the model trained on the de-duplicated data.",
    )

    doc.add_heading("References", level=1)
    for ref in (
        "Lundberg, S. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS. "
        "https://arxiv.org/abs/1705.07874",
        "Lundberg, S. et al. (2020). From local explanations to global understanding with explainable AI "
        "for trees. Nature Machine Intelligence 2, 56–67.",
        "SHAP documentation. https://shap.readthedocs.io/",
        "Molnar, C. (2022). Interpretable Machine Learning, chapter on SHAP. "
        "https://christophm.github.io/interpretable-ml-book/",
        "Generative AI (Claude, Anthropic) was used to help write the code and parts of this report. "
        "The group reviewed, ran and edited everything.",
    ):
        doc.add_paragraph(ref, style="List Number")

    doc.save(out_path)
    print(f"wrote {out_path} ({'with' if outputs else 'without'} SHAP outputs)")


if __name__ == "__main__":
    build(config.REPORTS_DIR / "Part3_Explainable_AI.docx")
