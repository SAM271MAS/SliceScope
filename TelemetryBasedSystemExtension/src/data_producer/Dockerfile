FROM python:3.11-slim
RUN pip install kafka-python
COPY live_producer.py /app/live_producer.py
WORKDIR /app
CMD ["python", "-u", "live_producer.py"]

