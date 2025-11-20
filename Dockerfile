FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libglib2.0-0 \
    wget gnupg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=2.1.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=10000

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /app
COPY api/pyproject.toml api/poetry.lock* ./api/
RUN cd api && poetry install --no-interaction --no-ansi

COPY api ./api

WORKDIR /app/api

# Render uses PORT environment variable (defaults to 10000)
EXPOSE $PORT
CMD poetry run uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
