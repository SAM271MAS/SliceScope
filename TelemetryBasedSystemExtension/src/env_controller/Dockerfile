FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn
COPY env_controller.py .
EXPOSE 8000
CMD ["uvicorn", "env_controller:app", "--host", "0.0.0.0", "--port", "8000"]

