"""
src/registry/artifacts.py
=========================
Single source of truth for the 11 ML artifact categories (Phase 2 scope, Section 7).

Used by:
  * GET /artifacts (via ModelStore.get_artifact_manifest)
  * scripts/run_phase2_pipeline.py step 3, which verifies every category exists
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

MODEL_FILES = ["difficulty_classifier.pkl", "blooms_classifier.pkl", "ct_classifier.pkl"]


@dataclass(frozen=True)
class Artifact:
    number: int
    key: str
    name: str
    location: str
    check: Callable[[Path, str], tuple[bool, str]]


def _files(*paths: str) -> Callable[[Path, str], tuple[bool, str]]:
    def check(root: Path, _uri: str) -> tuple[bool, str]:
        missing = [p for p in paths if not (root / p).exists()]
        return (not missing, "missing: " + ", ".join(missing) if missing else "present")

    return check


def _glob(pattern: str, minimum: int = 1) -> Callable[[Path, str], tuple[bool, str]]:
    def check(root: Path, _uri: str) -> tuple[bool, str]:
        n = len(list(root.glob(pattern)))
        return (n >= minimum, f"{n} file(s) match {pattern}")

    return check


def _models(root: Path, _uri: str) -> tuple[bool, str]:
    missing = [m for m in MODEL_FILES if not (root / "models" / m).exists()]
    return (
        not missing,
        "missing: " + ", ".join(missing) if missing else f"{len(MODEL_FILES)} models present",
    )


def _vectorizer(root: Path, _uri: str) -> tuple[bool, str]:
    path = root / "models" / "difficulty_classifier.pkl"
    if not path.exists():
        return False, "difficulty model missing"
    import joblib

    model = joblib.load(path)
    pipe = model["pipeline"] if isinstance(model, dict) else model
    steps = dict(getattr(pipe, "named_steps", {}))
    vec = steps.get("tfidf")
    ok = vec is not None and hasattr(vec, "vocabulary_")
    return ok, (
        f"fitted TF-IDF vocabulary: {len(vec.vocabulary_)} terms"
        if ok
        else "no fitted TF-IDF step in pipeline"
    )


def _mlflow_runs(metric: str | None) -> Callable[[Path, str], tuple[bool, str]]:
    def check(_root: Path, uri: str) -> tuple[bool, str]:
        try:
            import mlflow

            mlflow.set_tracking_uri(uri)
            client = mlflow.tracking.MlflowClient()
            exps = [e.experiment_id for e in client.search_experiments()]
            runs = client.search_runs(exps, max_results=500) if exps else []
            if metric:
                runs = [r for r in runs if metric in r.data.metrics]
            return (
                len(runs) > 0,
                f"{len(runs)} MLflow run(s)" + (f" with {metric}" if metric else ""),
            )
        except Exception as e:  # pragma: no cover - environment dependent
            return False, f"MLflow unavailable: {e}"

    return check


def _registry_metadata(_root: Path, uri: str) -> tuple[bool, str]:
    try:
        import mlflow

        mlflow.set_tracking_uri(uri)
        client = mlflow.tracking.MlflowClient()
        versions = [
            v
            for rm in client.search_registered_models()
            for v in client.search_model_versions(f"name='{rm.name}'")
        ]
        tagged = [v for v in versions if "data_hash" in (v.tags or {})]
        return (
            len(tagged) > 0,
            f"{len(versions)} registered version(s), {len(tagged)} tagged with data_hash",
        )
    except Exception as e:  # pragma: no cover - environment dependent
        return False, f"MLflow unavailable: {e}"


ARTIFACTS: list[Artifact] = [
    Artifact(
        1,
        "dataset",
        "Dataset",
        "DVC-versioned Parquet in data/processed/ (+ data/splits/ train / RVCE held-out)",
        _glob("data/processed/*.parquet", 1),
    ),
    Artifact(
        2,
        "preprocessing_pipeline",
        "Preprocessing pipeline",
        "src/data/preprocess.py, src/data/build_splits.py",
        _files("src/data/preprocess.py", "src/data/build_splits.py"),
    ),
    Artifact(
        3, "trained_model", "Trained models", "models/*.pkl (difficulty, Bloom's, CT)", _models
    ),
    Artifact(
        4,
        "model_configuration",
        "Model configuration",
        "configs/model_config.yaml",
        _files("configs/model_config.yaml"),
    ),
    Artifact(
        5,
        "evaluation_results",
        "Evaluation results",
        "MLflow experiment metrics (incl. rvce_* held-out metrics)",
        _mlflow_runs("test_f1_macro"),
    ),
    Artifact(
        6,
        "training_logs",
        "Training logs",
        "MLflow run telemetry (params, metrics, plots)",
        _mlflow_runs(None),
    ),
    Artifact(
        7,
        "feature_engineering_artifacts",
        "Feature engineering",
        "src/data/feature_engineering.py",
        _files("src/data/feature_engineering.py"),
    ),
    Artifact(
        8,
        "tokenizer",
        "Tokenizer / vectorizer",
        "TF-IDF vectorizer saved inside each model pipeline",
        _vectorizer,
    ),
    Artifact(
        9,
        "model_metadata",
        "Model metadata",
        "MLflow registry version tags (data_hash, model_type, run_name)",
        _registry_metadata,
    ),
    Artifact(
        10,
        "deployment_artifacts",
        "Deployment artifacts",
        "Dockerfile, docker-compose.yml, .github/workflows/ci.yml",
        _files("Dockerfile", "docker-compose.yml", ".github/workflows/ci.yml"),
    ),
    Artifact(
        11,
        "monitoring_artifacts",
        "Monitoring artifacts",
        "monitoring/prometheus.yml, Grafana dashboard and provisioning",
        _files(
            "monitoring/prometheus.yml",
            "monitoring/grafana/dashboards/questioncraft.json",
            "monitoring/grafana/provisioning/datasources/datasource.yml",
            "monitoring/grafana/provisioning/dashboards/dashboard.yml",
        ),
    ),
]


def verify_all(root: Path | str = ".", tracking_uri: str = "sqlite:///mlflow.db") -> list[dict]:
    """Run every artifact check; returns one result dict per category."""
    root = Path(root)
    out = []
    for a in ARTIFACTS:
        ok, detail = a.check(root, tracking_uri)
        out.append(
            {
                "number": a.number,
                "key": a.key,
                "name": a.name,
                "location": a.location,
                "ok": ok,
                "detail": detail,
            }
        )
    return out
