"""
src/registry/model_manager.py
=============================
MLflow Model Registry management for QuestionCraft.

Implements model lifecycle management:
- Model registration and versioning
- Stage transitions (None → Staging → Production → Archived)
- Model comparison and promotion (quality gate enforcement)
- Blue-Green deployment support (swap production models)

Aligned with RVCE MLOps AI254TA curriculum:
- ML Artifacts tracking (all 11 artifact types)
- Model governance and compliance
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("questioncraft.registry")

# Registry model names (must match Phase 1 training scripts)
REGISTRY_MODELS = {
    "difficulty": "questioncraft_difficulty_classifier",
    "blooms": "questioncraft_blooms_classifier",
    "ct": "questioncraft_ct_classifier",
}


class ModelRegistryManager:
    """
    Manages the MLflow Model Registry lifecycle for QuestionCraft models.
    
    Supports Blue-Green deployment through stage transitions:
    - None: Newly registered model, not yet evaluated
    - Staging: Passed quality gates, ready for canary testing
    - Production: Serving live traffic
    - Archived: Previous production version, kept for rollback
    """

    def __init__(self, tracking_uri: str = "sqlite:///mlflow.db"):
        self.tracking_uri = tracking_uri
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            self._client = mlflow.tracking.MlflowClient()
        return self._client

    def list_registered_models(self) -> list[dict]:
        """List all registered models and their versions."""
        results = []
        try:
            for rm in self.client.search_registered_models():
                versions = []
                for v in rm.latest_versions:
                    versions.append({
                        "version": v.version,
                        "stage": v.current_stage,
                        "status": v.status,
                        "run_id": v.run_id,
                        "created": str(v.creation_timestamp),
                    })
                results.append({
                    "name": rm.name,
                    "latest_versions": versions,
                    "description": rm.description or "",
                })
        except Exception as e:
            logger.error("Failed to list registered models: %s", e)
        return results

    def get_model_info(self, model_key: str) -> dict | None:
        """Get detailed info for a specific model by its QuestionCraft key."""
        model_name = REGISTRY_MODELS.get(model_key)
        if not model_name:
            return None

        try:
            rm = self.client.get_registered_model(model_name)
            versions = []
            for v in rm.latest_versions:
                # Get run metrics
                run = self.client.get_run(v.run_id)
                versions.append({
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "metrics": run.data.metrics,
                    "params": run.data.params,
                    "created": str(v.creation_timestamp),
                })
            return {
                "name": rm.name,
                "key": model_key,
                "versions": versions,
                "description": rm.description or "",
            }
        except Exception as e:
            logger.error("Failed to get model info for '%s': %s", model_key, e)
            return None

    def promote_to_staging(self, model_key: str, version: str | None = None) -> dict:
        """
        Promote a model version to Staging stage.
        If version is None, promotes the latest version.
        """
        model_name = REGISTRY_MODELS.get(model_key)
        if not model_name:
            return {"error": f"Unknown model key: {model_key}"}

        try:
            if version is None:
                versions = self.client.get_latest_versions(model_name, stages=["None"])
                if not versions:
                    return {"error": f"No unassigned versions found for {model_name}"}
                version = versions[0].version

            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging",
            )
            logger.info("Promoted %s v%s to Staging", model_name, version)
            return {
                "model": model_name,
                "version": version,
                "new_stage": "Staging",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error("Failed to promote to Staging: %s", e)
            return {"error": str(e)}

    def promote_to_production(self, model_key: str, version: str | None = None,
                               archive_existing: bool = True) -> dict:
        """
        Promote a model version to Production stage.
        
        Implements Blue-Green deployment:
        1. If archive_existing=True, the current Production version is
           moved to Archived stage (Green → becomes backup).
        2. The specified version is moved to Production (Blue → becomes active).
        """
        model_name = REGISTRY_MODELS.get(model_key)
        if not model_name:
            return {"error": f"Unknown model key: {model_key}"}

        try:
            # Archive existing production version (Blue-Green swap)
            if archive_existing:
                prod_versions = self.client.get_latest_versions(model_name, stages=["Production"])
                for pv in prod_versions:
                    self.client.transition_model_version_stage(
                        name=model_name,
                        version=pv.version,
                        stage="Archived",
                    )
                    logger.info(
                        "Archived previous Production %s v%s (Blue-Green swap)",
                        model_name, pv.version,
                    )

            # If no version specified, use latest Staging version
            if version is None:
                staging_versions = self.client.get_latest_versions(model_name, stages=["Staging"])
                if not staging_versions:
                    return {"error": f"No Staging version found for {model_name}"}
                version = staging_versions[0].version

            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
            )
            logger.info("Promoted %s v%s to Production", model_name, version)
            return {
                "model": model_name,
                "version": version,
                "new_stage": "Production",
                "strategy": "blue-green",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error("Failed to promote to Production: %s", e)
            return {"error": str(e)}

    def rollback(self, model_key: str) -> dict:
        """
        Rollback to the most recent Archived version (Blue-Green rollback).
        Current Production → Archived, most recent Archived → Production.
        """
        model_name = REGISTRY_MODELS.get(model_key)
        if not model_name:
            return {"error": f"Unknown model key: {model_key}"}

        try:
            archived = self.client.get_latest_versions(model_name, stages=["Archived"])
            if not archived:
                return {"error": "No archived version available for rollback"}

            # Archive current production
            prod_versions = self.client.get_latest_versions(model_name, stages=["Production"])
            for pv in prod_versions:
                self.client.transition_model_version_stage(
                    name=model_name, version=pv.version, stage="Archived",
                )

            # Restore most recent archived
            rollback_version = archived[0].version
            self.client.transition_model_version_stage(
                name=model_name, version=rollback_version, stage="Production",
            )
            logger.info("Rolled back %s to v%s", model_name, rollback_version)
            return {
                "model": model_name,
                "rolled_back_to_version": rollback_version,
                "strategy": "blue-green-rollback",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error("Rollback failed: %s", e)
            return {"error": str(e)}

    def compare_versions(self, model_key: str, version_a: str, version_b: str) -> dict:
        """Compare metrics between two model versions for quality gate decisions."""
        model_name = REGISTRY_MODELS.get(model_key)
        if not model_name:
            return {"error": f"Unknown model key: {model_key}"}

        try:
            v_a = self.client.get_model_version(model_name, version_a)
            v_b = self.client.get_model_version(model_name, version_b)

            run_a = self.client.get_run(v_a.run_id)
            run_b = self.client.get_run(v_b.run_id)

            metrics_a = run_a.data.metrics
            metrics_b = run_b.data.metrics

            comparison = {}
            all_metrics = set(metrics_a.keys()) | set(metrics_b.keys())
            for metric in all_metrics:
                val_a = metrics_a.get(metric)
                val_b = metrics_b.get(metric)
                comparison[metric] = {
                    f"v{version_a}": val_a,
                    f"v{version_b}": val_b,
                    "winner": f"v{version_a}" if (val_a or 0) >= (val_b or 0) else f"v{version_b}",
                }

            return {
                "model": model_name,
                "comparison": comparison,
                "recommendation": f"v{version_a}" if sum(
                    1 for m in comparison.values() if m["winner"] == f"v{version_a}"
                ) > len(comparison) / 2 else f"v{version_b}",
            }
        except Exception as e:
            logger.error("Comparison failed: %s", e)
            return {"error": str(e)}
