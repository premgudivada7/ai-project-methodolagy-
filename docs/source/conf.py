"""Sphinx configuration for the RetailGenius churn project documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "RetailGenius churn prediction"
author = "EPITA AI PM group"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",  # pull documentation from docstrings
    "sphinx.ext.napoleon",  # Google-style docstrings
    "sphinx.ext.viewcode",  # links to highlighted source code
    "sphinx.ext.autosummary",
]
autodoc_mock_imports = ["mlflow", "shap", "xgboost"]
autodoc_member_order = "bysource"
napoleon_google_docstring = True

templates_path = ["_templates"]
exclude_patterns = []

try:
    import sphinx_rtd_theme  # noqa: F401

    html_theme = "sphinx_rtd_theme"
except ImportError:
    html_theme = "alabaster"
