FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=2.1.3 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

WORKDIR /app
COPY api/pyproject.toml api/poetry.lock* ./api/
RUN cd api && poetry install --no-interaction --no-ansi

COPY api ./api
COPY settings.py ./

WORKDIR /app/api
EXPOSE 8000
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
