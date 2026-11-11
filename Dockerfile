# =============================================================================
# QuestionCraft — Multi-Stage Production Dockerfile
# =============================================================================
# Stage 1: Builder    — install pinned runtime dependencies into /install
# Stage 2: Production — slim runtime image running Uvicorn on port 8000
#
# Deployment target for Phase 2 is Docker Compose (see docker-compose.yml).
# =============================================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt .
# CPU-only torch first (for sentence-transformers), then everything else from PyPI.
RUN pip install --no-cache-dir --prefix=/install \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --prefix=/install -r requirements-serving.txt

# ── Stage 2: Production Runtime ───────────────────────────────────────────────
FROM python:3.13-slim AS production

LABEL maintainer="RVCE MLOps Team"
LABEL description="QuestionCraft Phase 2 — FastAPI Inference Service"
LABEL version="2.0.0"

# libgomp is required by xgboost at runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install /usr/local

COPY src/ src/
COPY configs/ configs/
COPY models/ models/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    PORT=8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# One worker: the canary ratio and request metrics live in process memory,
# so several workers would each hold different values.
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
