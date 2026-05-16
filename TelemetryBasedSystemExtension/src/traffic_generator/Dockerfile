FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir requests
COPY heavy_user.py .
COPY medium_user.py .
COPY light_user.py .
