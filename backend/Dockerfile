# ─────────────────────────────────────────────────────────────────────────────
# FairLens — Dockerfile for Cloud Run
#
# Build:
#   docker build -t fairlens-api .
#
# Run locally (env vars from .env file):
#   docker run --env-file .env -p 8080:8080 fairlens-api
#
# Env vars (SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY) are injected
# at runtime via Cloud Run secrets — never baked into the image.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some Python packages (scipy, reportlab)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for Cloud Run best practice
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy all backend source files (fixed — no hyperlink formatting)
COPY --chown=appuser:appgroup \
    api.py \
    data_loader.py \
    data_auditor.py \
    data_validator.py \
    model_auditor.py \
    mitigation.py \
    report_generator.py \
    database.py \
    storage.py \
    gemini.py \
    run_all.py \
    ./

# Outputs directory — writable by the non-root user
RUN mkdir -p /app/outputs && chown appuser:appgroup /app/outputs

USER appuser

# Cloud Run always routes to port 8080
EXPOSE 8080

# Env var defaults — overridden at runtime by Cloud Run secrets
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check — Cloud Run pings / before accepting traffic
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/')" || exit 1

# Uvicorn — 1 worker per Cloud Run instance (scale-out, not scale-up)
CMD ["uvicorn", "api:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--log-level", "info", \
     "--timeout-keep-alive", "75"]