"""
src/serving/app.py
==================
FastAPI inference service for QuestionCraft Phase 2.

Provides production REST endpoints for:
- Single and batch question classification (difficulty, Bloom's, CT skills)
- Semantic duplicate detection
- Model health checks and registry info
- ML artifact manifest (RVCE evaluation requirement)
- Canary deployment traffic routing
- Prometheus metrics export

Deployment strategies implemented:
- Blue-Green: Load production + staging models simultaneously
- Canary: Route configurable % of traffic to staging model

Usage:
    uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from src.serving.model_loader import ModelStore
from src.serving.schemas import (
    ArtifactManifest,
    BatchClassificationResult,
    BatchDuplicateInput,
    BatchDuplicateResult,
    BatchQuestionInput,
    ClassificationResult,
    DuplicateCheckInput,
    DuplicateResult,
    HealthResponse,
    QuestionInput,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("questioncraft.serving")

# ── Globals ────────────────────────────────────────────────────────────────────
_start_time: float = time.time()
model_store: ModelStore | None = None

# ── Prometheus metrics (lightweight, no external dependency required) ──────────
_request_count: dict[str, int] = {}
_request_latency_sum: dict[str, float] = {}
_prediction_count: dict[str, int] = {}


def _load_serving_config() -> dict:
    """Load serving configuration."""
    config_path = Path("configs/serving_config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


# ── Lifespan ───────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global model_store, _start_time
    _start_time = time.time()

    serving_cfg = _load_serving_config()
    models_dir = serving_cfg.get("models_dir", "models")
    # Environment wins over the config file, so containers can point at the MLflow server
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI") or serving_cfg.get(
        "mlflow_tracking_uri", "sqlite:///mlflow.db"
    )
    canary_ratio = serving_cfg.get("canary_ratio", 0.0)

    model_store = ModelStore(models_dir=models_dir, mlflow_uri=mlflow_uri)
    model_store.load_all()
    model_store.set_canary_ratio(canary_ratio)

    logger.info("QuestionCraft API ready — models loaded in %.2fs", model_store.load_time)
    yield
    logger.info("QuestionCraft API shutting down.")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="QuestionCraft Inference API",
    description=(
        "Production REST API for automated question classification and "
        "semantic duplicate detection. Supports Blue-Green and Canary "
        "deployment strategies with MLflow Model Registry integration."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: request metrics ────────────────────────────────────────────────


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start

    endpoint = request.url.path
    _request_count[endpoint] = _request_count.get(endpoint, 0) + 1
    _request_latency_sum[endpoint] = _request_latency_sum.get(endpoint, 0.0) + elapsed

    return response


# ── Health & Info Endpoints ────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint (used by the Docker HEALTHCHECK and the CI smoke test)."""
    return HealthResponse(
        status="healthy" if model_store and model_store.is_loaded("difficulty") else "degraded",
        version="2.0.0",
        models_loaded=model_store.get_all_status() if model_store else {},
        mlflow_connected=model_store.mlflow_uri is not None if model_store else False,
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.get("/artifacts", response_model=ArtifactManifest, tags=["System"])
async def get_artifact_manifest():
    """Return complete ML artifact manifest (RVCE evaluation requirement)."""
    if not model_store:
        raise HTTPException(status_code=503, detail="Models not loaded")
    manifest_data = model_store.get_artifact_manifest()
    return ArtifactManifest(**manifest_data)


@app.get("/models/info", tags=["Registry"])
async def list_models():
    """Serving versions, registry stages and canary state for every model."""
    if not model_store:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "models": model_store.model_versions,
        "canary_models": model_store.canary_versions,
        "canary_ratio": model_store._canary_ratio,
        "registry_stages": _registry_stages(),
        "status": model_store.get_all_status(),
    }


@app.post("/models/reload", tags=["Registry"])
async def reload_models():
    """Reload models from the registry after a promotion or rollback (blue-green swap)."""
    if not model_store:
        raise HTTPException(status_code=503, detail="Models not loaded")
    before = model_store.model_versions
    model_store.reload()
    return {
        "before": before,
        "after": model_store.model_versions,
        "canary_models": model_store.canary_versions,
        "canary_ratio": model_store._canary_ratio,
    }


def _registry_stages() -> dict:
    """{model_key: {version: stage}} for versions not in stage None; {} if MLflow is unreachable."""
    if not model_store or not model_store.mlflow_uri:
        return {}
    try:
        from src.registry.model_manager import REGISTRY_MODELS, ModelRegistryManager

        client = ModelRegistryManager(model_store.mlflow_uri).client
        out = {}
        for key, name in REGISTRY_MODELS.items():
            try:
                vs = client.search_model_versions(f"name='{name}'")
            except Exception:
                continue
            out[key] = {str(v.version): v.current_stage for v in vs if v.current_stage != "None"}
        return out
    except Exception as e:  # pragma: no cover - environment dependent
        logger.warning("Registry stages unavailable: %s", e)
        return {}


# ── Prometheus Metrics Endpoint ────────────────────────────────────────────────


@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics():
    """Prometheus-compatible metrics export endpoint."""
    lines = [
        "# HELP questioncraft_requests_total Total number of requests per endpoint",
        "# TYPE questioncraft_requests_total counter",
    ]
    for endpoint, count in _request_count.items():
        lines.append(f'questioncraft_requests_total{{endpoint="{endpoint}"}} {count}')

    lines.extend(
        [
            "# HELP questioncraft_request_latency_seconds_total Total request latency per endpoint",
            "# TYPE questioncraft_request_latency_seconds_total counter",
        ]
    )
    for endpoint, total_latency in _request_latency_sum.items():
        lines.append(
            f'questioncraft_request_latency_seconds_total{{endpoint="{endpoint}"}} {total_latency:.6f}'
        )

    lines.extend(
        [
            "# HELP questioncraft_predictions_total Total predictions per model type",
            "# TYPE questioncraft_predictions_total counter",
        ]
    )
    for model_type, count in _prediction_count.items():
        lines.append(f'questioncraft_predictions_total{{model="{model_type}"}} {count}')

    lines.extend(
        [
            "# HELP questioncraft_models_loaded Whether each model is loaded (1=yes, 0=no)",
            "# TYPE questioncraft_models_loaded gauge",
        ]
    )
    if model_store:
        for name, loaded in model_store.get_all_status().items():
            lines.append(f'questioncraft_models_loaded{{model="{name}"}} {1 if loaded else 0}')

    uptime = time.time() - _start_time
    lines.extend(
        [
            "# HELP questioncraft_uptime_seconds Server uptime in seconds",
            "# TYPE questioncraft_uptime_seconds gauge",
            f"questioncraft_uptime_seconds {uptime:.2f}",
        ]
    )

    # Prometheus text exposition format (must be plain text, not a JSON-encoded string)
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


# ── Classification Endpoints ──────────────────────────────────────────────────


def _classify_single(question_text: str) -> ClassificationResult:
    """Run all classifiers on a single question."""
    if not model_store:
        raise HTTPException(status_code=503, detail="Models not loaded")

    result = ClassificationResult(
        question_text=question_text,
        difficulty="Unknown",
        difficulty_confidence=0.0,
        blooms_level="Unknown",
        blooms_confidence=0.0,
        ct_skills=[],
        ct_confidences={},
    )

    # ── Difficulty ──
    diff_model = model_store.get_model("difficulty")
    if diff_model:
        try:
            pred = diff_model.predict([question_text])[0]
            result.difficulty = pred
            if hasattr(diff_model, "predict_proba"):
                proba = diff_model.predict_proba([question_text])[0]
                result.difficulty_confidence = round(float(max(proba)), 4)
            else:
                result.difficulty_confidence = 1.0
            _prediction_count["difficulty"] = _prediction_count.get("difficulty", 0) + 1
        except Exception as e:
            logger.error("Difficulty prediction failed: %s", e)

    # ── Bloom's ──
    blooms_model = model_store.get_model("blooms")
    if blooms_model:
        try:
            pred = blooms_model.predict([question_text])[0]
            result.blooms_level = pred
            if hasattr(blooms_model, "predict_proba"):
                proba = blooms_model.predict_proba([question_text])[0]
                result.blooms_confidence = round(float(max(proba)), 4)
            else:
                result.blooms_confidence = 1.0
            _prediction_count["blooms"] = _prediction_count.get("blooms", 0) + 1
        except Exception as e:
            logger.error("Bloom's prediction failed: %s", e)

    # ── CT Skills ──
    ct_model = model_store.get_model("ct")
    if ct_model:
        try:
            clf = (
                ct_model["pipeline"]
                if isinstance(ct_model, dict) and "pipeline" in ct_model
                else ct_model
            )
            pred = clf.predict([question_text])[0]
            ct_labels = [
                "Decomposition",
                "Pattern Recognition",
                "Abstraction",
                "Algorithmic Thinking",
                "Evaluation & Debugging",
            ]
            if hasattr(pred, "__iter__") and not isinstance(pred, str):
                # Multi-label binary output
                result.ct_skills = [ct_labels[i] for i, v in enumerate(pred) if v == 1]
            else:
                result.ct_skills = [str(pred)]
            _prediction_count["ct"] = _prediction_count.get("ct", 0) + 1
        except Exception as e:
            logger.error("CT prediction failed: %s", e)

    return result


@app.post("/predict", response_model=ClassificationResult, tags=["Inference"])
async def predict_single(body: QuestionInput):
    """
    Classify a single question across all dimensions:
    difficulty (Easy/Medium/Hard), Bloom's taxonomy (L1-L6),
    and Computational Thinking skills (multi-label).
    """
    return _classify_single(body.question_text)


@app.post("/classify", response_model=ClassificationResult, tags=["Inference"])
async def classify_single(body: QuestionInput):
    """Alias for /predict — classify a single question."""
    return _classify_single(body.question_text)


@app.post("/batch/classify", response_model=BatchClassificationResult, tags=["Inference"])
async def classify_batch(body: BatchQuestionInput):
    """Classify a batch of questions (up to 100)."""
    start = time.time()
    results = [_classify_single(q.question_text) for q in body.questions]
    elapsed_ms = (time.time() - start) * 1000

    return BatchClassificationResult(
        results=results,
        model_versions=model_store.model_versions if model_store else {},
        inference_time_ms=round(elapsed_ms, 2),
    )


# ── Duplicate Detection ───────────────────────────────────────────────────────


@app.post("/dedup", response_model=DuplicateResult, tags=["Deduplication"])
async def check_duplicate(body: DuplicateCheckInput):
    """Check if two questions are semantic duplicates."""
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([body.question_a, body.question_b])
        sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

        return DuplicateResult(
            question_a=body.question_a,
            question_b=body.question_b,
            cosine_similarity=round(sim, 4),
            is_duplicate=sim >= 0.85,
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="sentence-transformers not available")


@app.post("/batch/dedup", response_model=BatchDuplicateResult, tags=["Deduplication"])
async def check_batch_duplicates(body: BatchDuplicateInput):
    """Check all pairwise combinations for semantic duplicates."""
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(body.questions)
        sim_matrix = cosine_similarity(embeddings)

        pairs = []
        n = len(body.questions)
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i][j])
                pairs.append(
                    DuplicateResult(
                        question_a=body.questions[i],
                        question_b=body.questions[j],
                        cosine_similarity=round(sim, 4),
                        is_duplicate=sim >= body.threshold,
                    )
                )

        return BatchDuplicateResult(
            pairs=pairs,
            threshold_used=body.threshold,
            total_pairs_checked=len(pairs),
        )
    except ImportError:
        raise HTTPException(status_code=503, detail="sentence-transformers not available")


# ── Deployment Strategy Endpoints ──────────────────────────────────────────────


@app.post("/deploy/canary", tags=["Deployment"])
async def set_canary_ratio(ratio: float = 0.1):
    """
    Set canary traffic ratio for Blue-Green/Canary deployment strategy.

    - ratio=0.0: All traffic goes to Production model (Blue-Green stable)
    - ratio=0.1: 10% traffic goes to Staging model (Canary release)
    - ratio=1.0: All traffic goes to Staging model (full cutover)
    """
    if not model_store:
        raise HTTPException(status_code=503, detail="Models not loaded")
    model_store.set_canary_ratio(ratio)
    return {
        "message": f"Canary ratio set to {ratio * 100:.1f}%",
        "strategy": (
            "canary"
            if 0 < ratio < 1
            else ("blue-green-cutover" if ratio == 1 else "production-only")
        ),
    }


@app.get("/deploy/status", tags=["Deployment"])
async def deployment_status():
    """Get current deployment strategy status."""
    if not model_store:
        raise HTTPException(status_code=503, detail="Models not loaded")

    canary_ratio = model_store._canary_ratio
    has_canary = bool(model_store._canary_models)

    if canary_ratio == 0:
        strategy = "production-only"
    elif canary_ratio == 1:
        strategy = "blue-green-cutover"
    else:
        strategy = f"canary-{canary_ratio*100:.0f}pct"

    return {
        "active_strategy": strategy,
        "canary_ratio": canary_ratio,
        "canary_models_available": has_canary,
        "production_models": model_store.model_versions,
    }
