# Container for serving the champion model (bonus: cloud deployment).
# Build after `make all`, which creates mlflow.db + mlruns/:
#   docker build -t retailgenius-churn .
#   docker run -p 5001:5001 retailgenius-churn
# The same image can be pushed to Azure Container Apps, Google Cloud Run or AWS App Runner.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY churn/ churn/
COPY mlflow.db .
COPY mlruns/ mlruns/

ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db
EXPOSE 5001
CMD ["mlflow", "models", "serve", "-m", "models:/retailgenius-churn-model@champion", \
     "-h", "0.0.0.0", "-p", "5001", "--env-manager", "local"]
