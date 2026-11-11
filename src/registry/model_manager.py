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
                    versions.append(
                        {
                            "version": v.version,
                            "stage": v.current_stage,
                            "status": v.status,
                            "run_id": v.run_id,
                            "created": str(v.creation_timestamp),
                        }
                    )
                results.append(
                    {
                        "name": rm.name,
                        "latest_versions": versions,
                        "description": rm.description or "",
                    }
                )
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
                versions.append(
                    {
                        "version": v.version,
                        "stage": v.current_stage,
                        "run_id": v.run_id,
                        "metrics": run.data.metrics,
                        "params": run.data.params,
                        "created": str(v.creation_timestamp),
                    }
                )
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

    def promote_to_production(
        self, model_key: str, version: str | None = None, archive_existing: bool = True
    ) -> dict:
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
                        model_name,
                        pv.version,
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
                    name=model_name,
                    version=pv.version,
                    stage="Archived",
                )

            # Restore most recent archived
            rollback_version = archived[0].version
            self.client.transition_model_version_stage(
                name=model_name,
                version=rollback_version,
                stage="Production",
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
                "recommendation": (
                    f"v{version_a}"
                    if sum(1 for m in comparison.values() if m["winner"] == f"v{version_a}")
                    > len(comparison) / 2
                    else f"v{version_b}"
                ),
            }
        except Exception as e:
            logger.error("Comparison failed: %s", e)
            return {"error": str(e)}

    # ── Helpers used by the promotion CLI, pipeline and demo ───────────────────

    def get_stage_version(self, model_key: str, stage: str) -> Any | None:
        """Latest registry version currently in `stage` (or None)."""
        model_name = REGISTRY_MODELS.get(model_key)
        versions = self.client.get_latest_versions(model_name, stages=[stage])
        return versions[0] if versions else None

    def get_run_metrics_params(self, model_key: str, version: str) -> tuple[dict, dict]:
        """Metrics and params of the MLflow run that produced a registry version."""
        mv = self.client.get_model_version(REGISTRY_MODELS[model_key], str(version))
        run = self.client.get_run(mv.run_id)
        return dict(run.data.metrics), dict(run.data.params)

    def evaluate_promotion(self, model_key: str, version: str | None = None) -> dict:
        """
        Run ALL 6 quality gates for a candidate version against the current Production one.

        Candidate defaults to the Staging version. Metrics, data hashes and the
        regression baseline all come from the real MLflow runs.
        """
        from src.monitoring.quality_gates import run_full_quality_check

        if version is None:
            staged = self.get_stage_version(model_key, "Staging")
            if staged is None:
                return {"error": f"No Staging version for '{model_key}'"}
            version = staged.version
        metrics, params = self.get_run_metrics_params(model_key, version)
        metrics["test_size_n"] = float(params.get("test_size_n", 0))

        prod = self.get_stage_version(model_key, "Production")
        prod_metrics, prod_params = ({}, {})
        if prod is not None:
            prod_metrics, prod_params = self.get_run_metrics_params(model_key, prod.version)

        result = run_full_quality_check(
            metrics,
            production_metrics=prod_metrics or None,
            data_hash=params.get("data_hash"),
            expected_data_hash=prod_params.get("data_hash") or params.get("data_hash"),
        )
        result["candidate_version"] = str(version)
        result["production_version"] = prod.version if prod is not None else None
        result["candidate_metrics"] = metrics
        return result

    def set_demo_state(self, model_key: str, production: str, staging: str) -> None:
        """Put the registry in a known state: `production` live, `staging` as candidate."""
        name = REGISTRY_MODELS[model_key]
        for v in self.client.search_model_versions(f"name='{name}'"):
            target = {str(production): "Production", str(staging): "Staging"}.get(
                str(v.version), "None"
            )
            if v.current_stage != target:
                self.client.transition_model_version_stage(
                    name=name, version=v.version, stage=target
                )

    def latest_version_for_run(self, model_key: str, run_name: str) -> str | None:
        """Newest registry version whose source MLflow run has the given run name."""
        name = REGISTRY_MODELS[model_key]
        best = None
        for v in self.client.search_model_versions(f"name='{name}'"):
            try:
                rn = self.client.get_run(v.run_id).info.run_name
            except Exception:
                continue
            if rn == run_name and (best is None or int(v.version) > int(best)):
                best = str(v.version)
        return best

    def demo_versions(self, model_key: str = "difficulty") -> tuple[str, str]:
        """(production, staging) for the demo: newest TF-IDF+LR baseline vs newest XGBoost challenger."""
        prod = self.latest_version_for_run(model_key, "tfidf_lr_baseline")
        stag = self.latest_version_for_run(model_key, "xgboost_improved")
        if not prod or not stag:
            raise RuntimeError(
                "Need both a tfidf_lr_baseline and an xgboost_improved run registered"
            )
        return prod, stag

    def tag_version(self, model_key: str, version: str) -> dict:
        """Copy run metadata onto the registry version: data hash, model type, run name, key metrics."""
        name = REGISTRY_MODELS[model_key]
        mv = self.client.get_model_version(name, str(version))
        run = self.client.get_run(mv.run_id)
        tags = {
            "data_hash": run.data.params.get("data_hash", "unknown"),
            "data_file": run.data.params.get("data_file", "unknown"),
            "model_type": run.data.params.get(
                "model_type", "logistic_regression" if "lr_C" in run.data.params else "unknown"
            ),
            "run_name": run.info.run_name or "",
            "seed": run.data.params.get("seed", ""),
        }
        for metric in (
            "test_accuracy",
            "test_f1_macro",
            "test_f1_micro",
            "rvce_accuracy",
            "rvce_f1_micro",
        ):
            if metric in run.data.metrics:
                tags[metric] = f"{run.data.metrics[metric]:.4f}"
        for k, v in tags.items():
            self.client.set_model_version_tag(name, str(version), k, str(v))
        return tags

    def sync_tags(self) -> int:
        """Tag every registered version of every QuestionCraft model; returns how many were tagged."""
        n = 0
        for key, name in REGISTRY_MODELS.items():
            try:
                versions = self.client.search_model_versions(f"name='{name}'")
            except Exception:
                continue
            for v in versions:
                try:
                    self.tag_version(key, v.version)
                    n += 1
                except Exception as e:
                    logger.warning("Could not tag %s v%s: %s", name, v.version, e)
        return n
