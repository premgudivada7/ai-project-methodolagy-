# RetailGenius – Customer Churn Prediction

EPITA International Programs · AI Project Methodology 2026 · Graded project

**Group 5:** Aravind Reddy Kolan, Prem Siva Naga Sai Ram Kumar Gudivada, Praveen Kumar Koppula

This repository holds the technical parts of our group project:

* **Part 2:** a churn prediction model built with production-style practices: Git, a
  cookiecutter layout, PEP8, poetry/conda, Sphinx and MLflow (tracking, projects, registry
  and serving).
* **Part 3:** explaining the model with SHAP.

The reports are in [`reports/`](reports/): `Part1_Functional_Framing`, `Part2_Technical_Methodologies` (.docx + .pdf) and `Part3_Explainable_AI.docx` (rebuilt by `python scripts/build_xai_report.py` after `make explain`).

## Project structure

```
├── MLproject              <- MLflow Project: entry points of the pipeline
├── conda.yaml             <- environment used by `mlflow run`
├── pyproject.toml         <- poetry dependencies + black / pylint settings
├── requirements*.txt      <- same dependencies for pip users
├── Makefile               <- shortcuts: make data / experiments / register / explain / serve ...
├── Dockerfile             <- serving image (cloud deployment bonus)
├── data
│   ├── raw                <- original Kaggle file (E Commerce Dataset.xlsx), never modified
│   ├── interim            <- cleaned data and train / test split
│   └── processed          <- model-ready features
├── churn                  <- source code (python package)
│   ├── config.py          <- paths, columns, MLflow settings
│   ├── dataset.py         <- 1. data preparation
│   ├── features.py        <- 2. feature engineering + preprocessing pipeline
│   ├── train.py           <- 3. training + MLflow tracking
│   ├── experiments.py     <-    several runs to compare in the MLflow UI
│   ├── registry.py        <- 4. MLflow Model Registry (alias "champion")
│   ├── predict.py         <- 5. batch inference
│   └── explain.py         <- 6. SHAP explanations (Part 3)
├── notebooks              <- 01_eda.ipynb: exploratory analysis
├── scripts                <- query_server.py: test client for the REST server
├── tests                  <- pytest unit tests
├── docs                   <- Sphinx documentation (make docs)
├── models                 <- local copies of trained models (.joblib)
├── mlflow.db / mlruns     <- MLflow tracking store and artifacts (created on first run)
└── reports                <- reports (Part 1, 2, 3), figures, metrics, SHAP outputs
```

## Setup

The whole group uses **Python 3.11** (see `.python-version`).

```bash
# Option A: poetry
poetry install && poetry shell

# Option B: conda
conda env create -f conda.yaml && conda activate retailgenius-churn
pip install -r requirements-dev.txt
```

Put `E Commerce Dataset.xlsx` in `data/raw/` (it's already there in this repo).

## Running the pipeline

```bash
make experiments   # prepare data, build features, train 7 models -> MLflow
make ui            # http://127.0.0.1:5000 to compare the runs
make register      # best run (by cross-validated PR-AUC) -> registry, alias @champion
make explain       # SHAP figures in reports/figures/shap
make predict       # scores in reports/predictions.csv
```

Or run everything as an **MLflow Project**:

```bash
mlflow run . -e main --env-manager local
mlflow run . -e train -P model=gb -P n_estimators=300 --env-manager local
```

## Serving the model

```bash
make serve         # terminal 1: REST API on http://127.0.0.1:5001
make query         # terminal 2: sends 3 customers, prints churn probabilities
```

Example request:

```bash
curl -X POST http://127.0.0.1:5001/invocations -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
```

For the cloud bonus, build the Docker image after `make all` with `docker build -t retailgenius-churn .`.
The image runs as-is on Azure Container Apps, Google Cloud Run or AWS App Runner.

## Code quality

```bash
make format  # black
make lint    # black --check, flake8, pylint
make test    # pytest
make docs    # Sphinx HTML in docs/build/html
```

GitHub Actions runs black, flake8 and pytest on every push (`.github/workflows/ci.yml`).

## Results (test set, 1,015 customers)

| Model | CV PR-AUC | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|---|
| Random forest, 500 trees, full depth (champion) | 0.910 | 0.985 | 0.952 | 0.774 | 0.963 | 0.858 |
| Gradient boosting, 400 trees, depth 5 | 0.908 | 0.980 | 0.955 | 0.857 | 0.929 | 0.892 |
| Random forest, 300 trees, depth 12 | 0.886 | 0.978 | 0.927 | 0.756 | 0.914 | 0.827 |
| Logistic regression (baseline) | 0.730 | 0.888 | 0.690 | 0.839 | 0.465 | 0.599 |

The full table is in `reports/experiments_comparison.csv`.

## Data

The data is the [E-Commerce Customer Churn dataset](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)
from Kaggle: 5,630 customers, 20 columns, about 17% churners.
