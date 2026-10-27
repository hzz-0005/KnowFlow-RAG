FROM python:3.12-slim@sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
# PDFium needs a CJK fallback for PDFs that name a standard CID font without embedding it.
RUN apt-get update && apt-get install -y --no-install-recommends fonts-noto-cjk fontconfig \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.13.5
ENV UV_PROJECT_ENVIRONMENT=/opt/knowflow-venv \
    PATH="/opt/knowflow-venv/bin:$PATH"
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY LICENSE ./
COPY backend ./backend
COPY migrations ./migrations
COPY evaluation ./evaluation
COPY alembic.ini ./
RUN uv sync --frozen --no-dev --no-editable

EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn knowflow.app.main:app --host 0.0.0.0 --port 8000"]

