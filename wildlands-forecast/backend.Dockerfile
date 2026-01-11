# ----- backend.Dockerfile -----
FROM python:3.11-slim

# Systeemdeps (curl voor healthcheck, libgomp1 voor LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgomp1 build-essential gcc g++ \
 && rm -rf /var/lib/apt/lists/*

# Python deps
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir gunicorn

# App code
COPY . /app

# Run vanuit de backend-map
WORKDIR /app/backend
ENV PYTHONUNBUFFERED=1
EXPOSE 5000

# Start met gunicorn (Flask app heet "app" in app.py -> app:app)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "120", "app:app"]
