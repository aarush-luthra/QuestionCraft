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
        self._canary_versions: dict[str, str] = {}
        self._canary_ratio: float = 0.0  # 0.0 = all production, 1.0 = all canary
        self._load_time: float = 0.0

    @property
    def load_time(self) -> float:
        return self._load_time

    @property
    def canary_versions(self) -> dict[str, str]:
        return dict(self._canary_versions)

    def route(self, name: str) -> tuple[Any, str]:
        """Like get_model(), but also returns the version string that served the request."""
        if (
            self._canary_ratio > 0.0
            and name in self._canary_models
            and np.random.random() < self._canary_ratio
        ):
            return self._canary_models[name], self._canary_versions.get(name, "canary")
        return self._models.get(name), self._model_versions.get(name, "unknown")

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

    @staticmethod
    def _load_registry_model(mlflow, uri: str):
        """Load a registered sklearn pipeline (same interface as the local .pkl files)."""
        # Models trained via `python -m ...` were pickled under ``__main__``;
        # expose the custom estimator there so they can be unpickled here.
        import __main__
        from src.models.difficulty_classifier import LabelEncodedXGBClassifier

        if not hasattr(__main__, "LabelEncodedXGBClassifier"):
            __main__.LabelEncodedXGBClassifier = LabelEncodedXGBClassifier
        return mlflow.sklearn.load_model(uri)

    def _load_from_registry(self) -> None:
        """Attempt to load production-stage models from MLflow Model Registry."""
        try:
            import mlflow
            import mlflow.sklearn

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
                        self._models[key] = self._load_registry_model(mlflow, model_uri)
                        self._model_versions[key] = f"registry-v{version.version}-Production"
                        logger.info(
                            "Loaded '%s' from registry: %s v%s (Production)",
                            key,
                            model_name,
                            version.version,
                        )

                    # Also check for staging version (for canary deployment)
                    staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
                    if staging_versions:
                        staging_uri = f"models:/{model_name}/Staging"
                        self._canary_models[key] = self._load_registry_model(mlflow, staging_uri)
                        self._canary_versions[key] = (
                            f"registry-v{staging_versions[0].version}-Staging"
                        )
                        logger.info(
                            "Loaded canary '%s' from registry: %s v%s (Staging)",
                            key,
                            model_name,
                            staging_versions[0].version,
                        )
                except Exception as e:
                    logger.debug("Registry model '%s' not available: %s", model_name, e)

        except ImportError:
            logger.warning("MLflow not available — using local models only.")
        except Exception as e:
            logger.warning("Could not connect to MLflow registry: %s", e)

    def reload(self) -> None:
        """Re-read models (local + registry Production/Staging), e.g. after a promotion or rollback.

        The canary ratio is kept; if the registry no longer has a Staging version the canary
        routes nothing, because there is no canary model to route to.
        """
        self._models.clear()
        self._model_versions.clear()
        self._canary_models.clear()
        self._canary_versions.clear()
        self.load_all()

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
        return {name: name in self._models for name in ["difficulty", "blooms", "ct"]}

    def get_artifact_manifest(self) -> dict:
        """The 11 ML artifact categories (Phase 2 scope, Section 7) with their locations."""
        from src.registry.artifacts import ARTIFACTS

        manifest = {a.key: a.location for a in ARTIFACTS}
        manifest["trained_model"] += f" — loaded: {list(self._models.keys())}"
        manifest["model_metadata"] += f" — serving versions: {self._model_versions}"
        manifest["training_logs"] += " — tracking URI: " + (
            self.mlflow_uri or "sqlite:///mlflow.db"
        )
        return manifest
