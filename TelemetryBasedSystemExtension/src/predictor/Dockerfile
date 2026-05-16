FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir \
    tensorflow==2.13.0 \
    pandas==2.0.3 \
    scikit-learn==1.8.0 \
    influxdb==5.3.1 \
    kfp==2.0.1 \
    joblib==1.3.2 \
    boto3==1.28.38

COPY predictor.py .
CMD ["python", "-u", "predictor.py"]
