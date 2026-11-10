# =============================================================================
# QuestionCraft — Multi-Stage Production Dockerfile
# =============================================================================
# Stage 1: Builder — install dependencies
# Stage 2: Production — lean runtime image
#
# Implements containerization for RVCE MLOps AI254TA curriculum
# Compatible with Kubernetes deployment (k8s/ manifests)
# =============================================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System dependencies for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency spec and install
COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install \
    fastapi==0.111.0 \
    uvicorn[standard]==0.30.1 \
    pydantic==2.7.4 \
    scikit-learn==1.4.2 \
    xgboost==2.0.3 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scipy==1.13.0 \
    joblib==1.4.2 \
    pyyaml==6.0.1 \
    mlflow==2.13.0 \
    nltk==3.8.1 \
    textstat==0.7.3 \
    sentence-transformers==2.7.0 \
    torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu \
    evidently==0.4.22

# ── Stage 2: Production Runtime ───────────────────────────────────────────────
FROM python:3.11-slim AS production

LABEL maintainer="RVCE MLOps Team"
LABEL description="QuestionCraft Phase 2 — FastAPI Inference Service"
LABEL version="2.0.0"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY models/ models/

# Download NLTK data
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng stopwords

# Environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
    PORT=8000

# Health check for Kubernetes liveness probe
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# Run with Uvicorn
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
