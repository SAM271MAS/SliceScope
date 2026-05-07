FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy pandas scikit-learn tensorflow optuna minio kfp

RUN pip install --no-cache-dir influxdb-client python-dotenv

RUN pip install --no-cache-dir matplotlib
RUN pip install --no-cache-dir mlflow
RUN pip install --no-cache-dir prometheus-client
COPY . .

ENV PYTHONUNBUFFERED=1
