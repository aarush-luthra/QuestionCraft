"""
scripts/run_phase2_pipeline.py
==============================
Phase 2 Productionization Orchestrator for QuestionCraft.

Runs the complete Phase 2 pipeline:
1. Model quality gate validation
2. Data validation (rule-based checks)
3. ML artifact verification (all 11 categories)
4. FastAPI server launch with model loading

Usage:
    python scripts/run_phase2_pipeline.py
    python scripts/run_phase2_pipeline.py --skip-server
    python scripts/run_phase2_pipeline.py --demo-only
"""

from __future__ import annotations

import json
import logging
import subprocess
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
logger = logging.getLogger("phase2_pipeline")


def banner(text: str) -> None:
    width = max(60, len(text) + 4)
    logger.info("=" * width)
    logger.info(f"  {text}")
    logger.info("=" * width)


def step_1_validate_models() -> bool | None:
    """Run all 6 model quality gates using real metrics from the MLflow registry."""
    banner("STEP 1: Model Quality Gate Validation")
    try:
        from src.registry.model_manager import ModelRegistryManager

        manager = ModelRegistryManager()
        if not manager.list_registered_models():
            logger.warning(
                "No models in the MLflow registry — skipping quality gates (run Phase 1 training first)"
            )
            return None
        prod_v, stag_v = manager.demo_versions("difficulty")
        manager.set_demo_state("difficulty", production=prod_v, staging=stag_v)
        result = manager.evaluate_promotion("difficulty")
        if "error" in result:
            logger.error("Quality gates could not run: %s", result["error"])
            return False
        logger.info(
            "Candidate v%s (Staging) vs Production v%s",
            result["candidate_version"],
            result["production_version"],
        )
        for g in result["details"]:
            logger.info("  [%s] %s: %s", "PASS" if g["passed"] else "FAIL", g["gate"], g["message"])
        m = result["candidate_metrics"]
        if "rvce_accuracy" in m:
            logger.info(
                "  [INFO] RVCE held-out (real papers, never trained on): accuracy %.3f vs majority baseline %.3f (n=%d)",
                m["rvce_accuracy"],
                m.get("rvce_majority_baseline_accuracy", float("nan")),
                int(m.get("rvce_n", 0)),
            )

        if result["all_passed"]:
            logger.info(
                "✅ All %d quality gates PASSED — model eligible for promotion",
                result["total_gates"],
            )
        else:
            logger.warning(
                "❌ Quality gates FAILED — %d/%d gates failed",
                result["failed"],
                result["total_gates"],
            )
        return result["all_passed"]

    except Exception as e:
        logger.error("Quality gate validation failed: %s", e)
        return False


def step_2_validate_data() -> bool | None:
    """Run data validation suite."""
    banner("STEP 2: Data Validation (rule-based checks)")
    try:
        import pandas as pd

        from src.monitoring.model_tests import validate_difficulty_dataset

        data_file = Path("data/processed/difficulty_dataset.parquet")
        if not data_file.exists():
            logger.warning("Processed data not found at %s — skipping data validation", data_file)
            logger.info("(Run 'dvc repro' first to generate processed datasets)")
            return None

        from src.monitoring.model_tests import clean_text_dataset

        raw = pd.read_parquet(data_file)
        raw_result = validate_difficulty_dataset(raw)
        for d in raw_result["details"]:
            if not d["passed"]:
                logger.warning("  raw data issue — %s: %s", d["test"], d["message"])

        df, clean_report = clean_text_dataset(raw, "question_text", "difficulty")
        logger.info("Cleaning removed: %s", json.dumps(clean_report))
        result = validate_difficulty_dataset(df)
        logger.info(
            "Validation of cleaned data: %d/%d checks passed",
            result["passed"],
            result["total_tests"],
        )

        if result["all_passed"]:
            logger.info("✅ All data validation checks PASSED")
        else:
            logger.warning(
                "❌ Data validation FAILED — %d/%d checks failed",
                result["failed"],
                result["total_tests"],
            )
        return result["all_passed"]

    except Exception as e:
        logger.error("Data validation failed: %s", e)
        return False


def step_3_check_artifacts() -> bool:
    """Verify all 11 ML artifact categories exist (Phase 2 scope, Section 7)."""
    banner("STEP 3: ML Artifact Verification (11 categories)")
    from src.registry.artifacts import verify_all

    results = verify_all(".")
    for r in results:
        logger.info(
            "  %s %2d. %-24s %s", "✅" if r["ok"] else "❌", r["number"], r["name"], r["detail"]
        )
    ok = all(r["ok"] for r in results)
    if ok:
        logger.info("✅ All 11 ML artifact categories verified")
    else:
        logger.warning(
            "❌ %d/11 artifact categories missing — run Phase 1 training first",
            sum(not r["ok"] for r in results),
        )
    return ok


def step_4_demo_api(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launch the FastAPI server and run demo requests."""
    banner("STEP 4: FastAPI Inference Demo")

    logger.info("Starting FastAPI server on http://%s:%d ...", host, port)
    logger.info("Press Ctrl+C to stop the server")
    logger.info("")
    logger.info("Available endpoints:")
    logger.info("  GET  /health         — Health check (Docker HEALTHCHECK)")
    logger.info("  GET  /docs           — Swagger UI documentation")
    logger.info("  GET  /metrics        — Prometheus metrics")
    logger.info("  GET  /artifacts      — ML artifact manifest")
    logger.info("  GET  /models/info    — Model registry info")
    logger.info("  GET  /deploy/status  — Deployment strategy status")
    logger.info("  POST /predict        — Single question classification")
    logger.info("  POST /classify       — Alias for /predict")
    logger.info("  POST /batch/classify — Batch classification (up to 100)")
    logger.info("  POST /dedup          — Duplicate check (pair)")
    logger.info("  POST /batch/dedup    — Batch duplicate detection")
    logger.info("  POST /deploy/canary  — Set canary traffic ratio")
    logger.info("")
    logger.info("Example curl commands:")
    logger.info("  curl http://localhost:8000/health")
    logger.info(
        '  curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d \'{"question_text": "Design a B-Tree for key-value storage"}\''
    )
    logger.info("")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "src.serving.app:app",
                "--host",
                host,
                "--port",
                str(port),
                "--reload",
            ],
            check=True,
        )
    except KeyboardInterrupt:
        logger.info("Server stopped.")
    except FileNotFoundError:
        logger.error("uvicorn not found — install with: pip install uvicorn")


@click.command()
@click.option("--skip-server", is_flag=True, help="Run validations only, don't start the server")
@click.option("--strict", is_flag=True, help="Exit non-zero if any check fails (for CI)")
@click.option("--demo-only", is_flag=True, help="Skip validations, start server directly")
@click.option("--host", default="127.0.0.1", help="Server host")
@click.option("--port", default=8000, help="Server port")
def main(skip_server: bool, strict: bool, demo_only: bool, host: str, port: int) -> None:
    banner("QuestionCraft Phase 2 — Productionization Pipeline")
    logger.info("")

    if not demo_only:
        results = {
            "quality_gates": step_1_validate_models(),
            "data_validation": step_2_validate_data(),
            "artifacts": step_3_check_artifacts(),
        }

        banner("Pipeline Summary")
        for step, passed in results.items():
            status = {True: "✅ PASSED", False: "❌ FAILED", None: "⏭️  SKIPPED"}[passed]
            logger.info("  %s: %s", step, status)

        # A skipped check is not a pass: --strict fails on it too.
        all_passed = all(v is True for v in results.values())
        logger.info("")
        if all_passed:
            logger.info("🎉 All Phase 2 checks passed — ready for deployment!")
        else:
            logger.warning("⚠️  Some checks failed — review above for details")
            if strict:
                raise SystemExit(1)

    if not skip_server:
        step_4_demo_api(host, port)


if __name__ == "__main__":
    main()
