# --- Builder ---
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# --- Runtime ---
FROM python:3.13-slim

WORKDIR /app

COPY --from=builder /app/.venv .venv
COPY main.py config.py ./
COPY api/ api/
COPY model/ model/

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_PATH="model/modelo_cancer_v1.joblib"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
