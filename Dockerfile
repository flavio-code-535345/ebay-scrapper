# ── Build stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    find /usr/local -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# ── Runtime stage ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local /usr/local

COPY app.py database.py scraper.py ebay_api_client.py ./
COPY ai_providers/ ai_providers/
COPY prompts/ prompts/
COPY templates/ templates/
COPY static/ static/

RUN mkdir -p /data

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_FORMAT=json

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--threads", "4", "--timeout", "180", "--access-logfile", "-", "--error-logfile", "-", "app:app"]

