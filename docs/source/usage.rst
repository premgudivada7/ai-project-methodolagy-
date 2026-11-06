Usage
=====

Pipeline steps
--------------

.. code-block:: bash

   make data          # churn.dataset   - cleaning + train/test split
   make features      # churn.features  - feature engineering
   make experiments   # churn.experiments - several models logged in MLflow
   make register      # churn.registry  - best model -> Model Registry (@champion)
   make explain       # churn.explain   - SHAP plots
   make serve         # local REST inference server
   make query         # test request to the server
   make docs          # this documentation

The same steps can be run as an MLflow Project::

   mlflow run . -e main --env-manager local
