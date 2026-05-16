FROM python:3.11-slim
RUN pip install kafka-python influxdb
COPY kafka_to_influx.py /app/kafka_to_influx.py
WORKDIR /app
CMD ["python", "-u", "kafka_to_influx.py"]

