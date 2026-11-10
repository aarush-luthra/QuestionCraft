"""
src/serving/model_loader.py
===========================
Model loader supporting both local .pkl files and MLflow Model Registry.

Supports Blue-Green deployment by allowing two model versions to be loaded
simultaneously (production + staging) for canary traffic routing.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger("questioncraft.serving.model_loader")


class ModelStore:
    """
    Central model store that loads and manages all QuestionCraft models.
    
    Supports:
    - Local .pkl file loading (default, always works)
    - MLflow Model Registry loading (production deployment)
    - Blue-Green model swapping (load new version alongside old)
    - Canary routing (percentage-based traffic splitting between versions)
    """

    def __init__(self, models_dir: str = "models", mlflow_uri: str | None = None):
        self.models_dir = Path(models_dir)
        self.mlflow_uri = mlflow_uri
        self._models: dict[str, Any] = {}
        self._model_versions: dict[str, str] = {}
        self._canary_models: dict[str, Any] = {}  # staging/canary versions
        self._canary_ratio: float = 0.0  # 0.0 = all production, 1.0 = all canary
        self._load_time: float = 0.0

    @property
    def load_time(self) -> float:
        return self._load_time

    @property
    def model_versions(self) -> dict[str, str]:
        return dict(self._model_versions)

    def load_all(self) -> None:
        """Load all models from local pkl files."""
        start = time.time()
        model_files = {
            "difficulty": "difficulty_classifier.pkl",
            "blooms": "blooms_classifier.pkl",
            "ct": "ct_classifier.pkl",
        }

        for name, filename in model_files.items():
            path = self.models_dir / filename
            if path.exists():
                try:
                    self._models[name] = joblib.load(path)
                    self._model_versions[name] = f"local-{path.stat().st_mtime_ns}"
                    logger.info("Loaded model '%s' from %s", name, path)
                except Exception as e:
                    logger.error("Failed to load model '%s': %s", name, e)
            else:
                logger.warning("Model file not found: %s", path)

        # Try loading from MLflow registry if configured
        if self.mlflow_uri:
            self._load_from_registry()

        self._load_time = time.time() - start
        logger.info(
            "Model store initialized in %.2fs — loaded: %s",
            self._load_time,
            list(self._models.keys()),
        )

    def _load_from_registry(self) -> None:
        """Attempt to load production-stage models from MLflow Model Registry."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self.mlflow_uri)
            registry_models = {
                "difficulty": "questioncraft_difficulty_classifier",
                "blooms": "questioncraft_blooms_classifier",
                "ct": "questioncraft_ct_classifier",
            }

            client = mlflow.tracking.MlflowClient()
            for key, model_name in registry_models.items():
                try:
                    # Try loading production version
                    versions = client.get_latest_versions(model_name, stages=["Production"])
                    if versions:
                        version = versions[0]
                        model_uri = f"models:/{model_name}/Production"
                        self._models[key] = mlflow.pyfunc.load_model(model_uri)
                        self._model_versions[key] = f"registry-v{version.version}-Production"
                        logger.info(
                            "Loaded '%s' from registry: %s v%s (Production)",
                            key, model_name, version.version,
                        )

                    # Also check for staging version (for canary deployment)
                    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
                    if staging_versions:
                        staging_uri = f"models:/{model_name}/Staging"
                        self._canary_models[key] = mlflow.pyfunc.load_model(staging_uri)
                        logger.info(
                            "Loaded canary '%s' from registry: %s v%s (Staging)",
                            key, model_name, staging_versions[0].version,
                        )
                except Exception as e:
                    logger.debug("Registry model '%s' not available: %s", model_name, e)

        except ImportError:
            logger.warning("MLflow not available — using local models only.")
        except Exception as e:
            logger.warning("Could not connect to MLflow registry: %s", e)

    def set_canary_ratio(self, ratio: float) -> None:
        """Set canary traffic ratio (0.0 = all production, 1.0 = all canary/staging)."""
        self._canary_ratio = max(0.0, min(1.0, ratio))
        logger.info("Canary ratio set to %.1f%%", self._canary_ratio * 100)

    def get_model(self, name: str) -> Any:
        """
        Get a model by name, respecting canary routing.
        
        If canary is configured and a staging model exists, routes
        traffic based on the configured canary ratio.
        """
        if (
            self._canary_ratio > 0.0
            and name in self._canary_models
            and np.random.random() < self._canary_ratio
        ):
            logger.debug("Routing to canary model for '%s'", name)
            return self._canary_models[name]
        return self._models.get(name)

    def is_loaded(self, name: str) -> bool:
        return name in self._models

    def get_all_status(self) -> dict[str, bool]:
        return {
            name: name in self._models
            for name in ["difficulty", "blooms", "ct"]
        }

    def get_artifact_manifest(self) -> dict:
        """Return complete ML artifact manifest as required by RVCE evaluation."""
        return {
            "dataset": "data/processed/ (DVC-versioned Parquet files: difficulty_dataset, blooms_dataset, ct_dataset)",
            "preprocessing_pipeline": "src/data/preprocess.py (normalization, feature extraction, label encoding)",
            "trained_model": f"models/ directory — loaded models: {list(self._models.keys())}",
            "model_configuration": "configs/model_config.yaml (pinned hyperparameters, TF-IDF settings, thresholds)",
            "evaluation_results": "MLflow experiments (accuracy, F1-macro, precision, recall, CV scores)",
            "training_logs": "MLflow tracking URI: " + (self.mlflow_uri or "sqlite:///mlflow.db"),
            "feature_engineering_artifacts": "src/data/feature_engineering.py (48 features: readability, POS, Bloom verbs, CT keywords)",
            "tokenizer": "TF-IDF vectorizer (max_features=20000, ngram_range=(1,2), sublinear_tf=True)",
            "model_metadata": f"Model versions: {self._model_versions}",
            "deployment_artifacts": "Dockerfile, docker-compose.yml, k8s/ manifests, configs/serving_config.yaml",
            "monitoring_artifacts": "Evidently AI reports, Prometheus metrics, Grafana dashboards",
        }
