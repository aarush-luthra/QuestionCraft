"""
scripts/promote_model.py
========================
Model promotion script for QuestionCraft.

Promotes models through the MLflow Model Registry lifecycle:
  None → Staging → Production → Archived

Implements quality gate enforcement before promotion.

Usage:
    python scripts/promote_model.py --model difficulty --to staging
    python scripts/promote_model.py --model difficulty --to production
    python scripts/promote_model.py --model difficulty --rollback
    python scripts/promote_model.py --list
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("promote_model")


@click.command()
@click.option("--model", type=click.Choice(["difficulty", "blooms", "ct"]), help="Model to promote")
@click.option("--to", "stage", type=click.Choice(["staging", "production"]), help="Target stage")
@click.option("--version", default=None, help="Specific version to promote")
@click.option("--rollback", is_flag=True, help="Rollback to previous production version")
@click.option("--list", "list_models", is_flag=True, help="List all registered models")
@click.option("--compare", nargs=2, type=str, help="Compare two versions: --compare v1 v2")
@click.option(
    "--sync-tags",
    is_flag=True,
    help="Copy run metadata (data hash, metrics) onto every registry version",
)
@click.option(
    "--skip-gates", is_flag=True, help="Promote to Production without running quality gates"
)
@click.option(
    "--tracking-uri",
    default=lambda: os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
    help="MLflow tracking URI (env MLFLOW_TRACKING_URI; use http://localhost:5001 while the Compose stack runs)",
)
def main(
    model: str | None,
    stage: str | None,
    version: str | None,
    rollback: bool,
    list_models: bool,
    compare: tuple[str, str] | None,
    sync_tags: bool,
    skip_gates: bool,
    tracking_uri: str,
) -> None:
    from src.registry.model_manager import ModelRegistryManager

    manager = ModelRegistryManager(tracking_uri=tracking_uri)

    if sync_tags:
        logger.info("Tagged %d registry versions", manager.sync_tags())
        return

    if list_models:
        logger.info("Registered models:")
        models = manager.list_registered_models()
        print(json.dumps(models, indent=2, default=str))
        return

    if not model:
        logger.error("--model is required (difficulty, blooms, or ct)")
        raise SystemExit(1)

    if compare:
        v_a, v_b = compare
        result = manager.compare_versions(model, v_a, v_b)
        print(json.dumps(result, indent=2, default=str))
        return

    if rollback:
        logger.info("Rolling back '%s' to previous production version...", model)
        result = manager.rollback(model)
        print(json.dumps(result, indent=2, default=str))
        return

    if not stage:
        logger.error("--to is required (staging or production)")
        raise SystemExit(1)

    if stage == "staging":
        logger.info("Promoting '%s' to Staging...", model)
        result = manager.promote_to_staging(model, version=version)
    else:
        if not skip_gates:
            logger.info("Running all quality gates for '%s' before Production promotion...", model)
            gate_result = manager.evaluate_promotion(model, version=version)
            if "error" in gate_result:
                print(json.dumps(gate_result, indent=2))
                raise SystemExit(1)
            for g in gate_result["details"]:
                logger.info(
                    "  [%s] %s: %s", "PASS" if g["passed"] else "FAIL", g["gate"], g["message"]
                )
            if not gate_result["promotion_allowed"]:
                logger.error(
                    "Quality gates FAILED — promotion BLOCKED (use --skip-gates to override)"
                )
                raise SystemExit(1)
        # Production promotion with Blue-Green swap
        logger.info("Promoting '%s' to Production (Blue-Green swap)...", model)
        result = manager.promote_to_production(model, version=version, archive_existing=True)

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
