.PHONY: install data features train experiments register explain predict serve query ui lint format test docs all clean

PYTHON ?= python
export MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db
PORT ?= 5001

install:            ## install the project with poetry (or: pip install -r requirements-dev.txt && pip install -e .)
	poetry install

data:               ## clean the raw data and split train / test
	$(PYTHON) -m churn.dataset

features: data      ## feature engineering
	$(PYTHON) -m churn.features

train: features     ## train the default model (random forest)
	$(PYTHON) -m churn.train --model rf --n-estimators 300 --max-depth 12 --run-name rf_default

experiments: features  ## train several models -> compare them in the MLflow UI
	$(PYTHON) -m churn.experiments

register:           ## register the best run in the Model Registry (alias: champion)
	$(PYTHON) -m churn.registry --metric cv_pr_auc_mean

explain:            ## SHAP explanations of the champion model (Part 3)
	$(PYTHON) -m churn.explain

predict:            ## batch inference on the test customers
	$(PYTHON) -m churn.predict

ui:                 ## MLflow tracking UI on http://127.0.0.1:5000
	mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --port 5000

serve:              ## local REST inference server for the champion model
	mlflow models serve -m "models:/retailgenius-churn-model@champion" -p $(PORT) --env-manager local

query:              ## send 3 test customers to the local server
	$(PYTHON) scripts/query_server.py --port $(PORT)

mlproject:          ## run the whole pipeline as an MLflow Project
	mlflow run . -e main --env-manager local

lint:
	black --check churn tests scripts
	flake8 churn tests scripts
	pylint churn --fail-under=8

format:
	black churn tests scripts

test:
	pytest -q

docs:               ## build the Sphinx HTML documentation in docs/build/html
	sphinx-build -b html docs/source docs/build/html

all: experiments register explain predict

clean:
	rm -rf data/interim/*.csv data/processed/*.csv docs/build
