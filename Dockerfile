# ── Build stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt && \
    find /root/.local -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# ── Runtime stage ──────────────────────────────────────────────────────────
FROM python:3.11-slim

RUN groupadd -r ebay && useradd -r -g ebay -d /app ebay

WORKDIR /app

COPY --from=builder /root/.local /home/ebay/.local

COPY app.py database.py gemini_assessor.py scraper.py ebay_api_client.py ./
COPY templates/ templates/
COPY static/ static/

RUN mkdir -p /data && chown ebay:ebay /data

ENV PATH=/home/ebay/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER ebay

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/api/health')" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "180", "app:app"]
