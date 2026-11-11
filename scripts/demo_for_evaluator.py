"""
scripts/demo_for_evaluator.py
=============================
Interactive Live Demo Script for RVCE MLOps Phase 2 Evaluation.
Demonstrates:
  1. Automated Data Validation & Pre-promotion Quality Gates (Unit III)
  2. Live Question Classification & Semantic Deduplication
  3. The 11 ML Artifacts Manifest (Accreditation & Auditability)
  4. Blue-Green & Canary Deployment Strategies with Rollback (Unit IV)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd


def _demo_versions() -> tuple[str, str]:
    """Newest TF-IDF+LR baseline (Production) and newest XGBoost challenger (Staging)."""
    from src.registry.model_manager import ModelRegistryManager

    return ModelRegistryManager().demo_versions("difficulty")


PRODUCTION_VERSION, STAGING_VERSION = _demo_versions()


def print_header(title: str, subtitle: str = ""):
    print("\n" + "=" * 75)
    print(f"  >>> {title.upper()} <<<")
    if subtitle:
        print(f"      {subtitle}")
    print("=" * 75)


def demo_step_1_quality_gates():
    print_header(
        "Step 1: Model Risk, Quality Gates & Data Validation",
        "RVCE AI254TA Unit III: Quality Assurance & Reproducibility",
    )
    from src.monitoring.model_tests import clean_text_dataset, validate_difficulty_dataset
    from src.registry.model_manager import ModelRegistryManager

    print("\n[+] Rule-based data validation on the RAW processed dataset...")
    raw = pd.read_parquet("data/processed/difficulty_dataset.parquet")
    raw_res = validate_difficulty_dataset(raw)
    print(
        f"    - Raw rows: {len(raw):,} | Checks passed: {raw_res['passed']}/{raw_res['total_tests']}"
    )
    for d in raw_res["details"]:
        if not d["passed"]:
            print(f"      * [FAILED] {d['test']}: {d['message']}")

    print("\n[+] Remediation: removing short texts, conflicting labels and exact duplicates...")
    clean, report = clean_text_dataset(raw, "question_text", "difficulty")
    print(
        f"    - {report['rows_before']:,} -> {report['rows_after']:,} rows  "
        f"(short: {report['removed_too_short']}, conflicting-label texts: "
        f"{report['removed_conflicting_label_texts']}, duplicates: {report['removed_duplicates']:,})"
    )
    clean_res = validate_difficulty_dataset(clean)
    print(
        f"    - Re-validation on cleaned data: {clean_res['passed']}/{clean_res['total_tests']} checks passed"
    )

    print("\n[+] Enforcing all 6 pre-promotion quality gates (real MLflow metrics)...")
    manager = ModelRegistryManager()
    manager.set_demo_state("difficulty", production=PRODUCTION_VERSION, staging=STAGING_VERSION)
    gate_res = manager.evaluate_promotion("difficulty")
    print(
        f"    - Candidate: v{gate_res['candidate_version']} (Staging, XGBoost) vs "
        f"Production: v{gate_res['production_version']} (Logistic Regression)"
    )
    print(f"    - Gates Passed: {gate_res['passed']}/{gate_res['total_gates']}")
    for g in gate_res["details"]:
        print(f"      * [{'PASS' if g['passed'] else 'FAIL'}] {g['gate']}: {g['message']}")
    m = gate_res["candidate_metrics"]
    if "rvce_accuracy" in m:
        print("\n[+] Domain-shift check on real RVCE papers (held out, never trained on):")
        print(
            f"    - Accuracy {m['rvce_accuracy']:.3f} vs majority-class baseline "
            f"{m.get('rvce_majority_baseline_accuracy', float('nan')):.3f} (n={int(m.get('rvce_n', 0))})"
        )
        print("    - Labels: LLM drafts pending faculty review -> reported, not gated")
    print(
        f"\n    ==> Model Promotion Gate Decision: "
        f"{'APPROVED (eligible for Production)' if gate_res['promotion_allowed'] else 'BLOCKED'}"
    )


def demo_step_2_live_inference():
    print_header(
        "Step 2: Live Multi-Dimensional Question Classification",
        "Production ML Inference (Difficulty + Bloom's + CT Skills)",
    )
    from src.serving.model_loader import ModelStore

    store = ModelStore(models_dir="models")
    store.load_all()

    test_questions = [
        "Define a binary search tree and state its properties.",
        "Apply Dijkstra's algorithm to find the shortest path from A to all nodes in the given graph.",
        "Evaluate the trade-offs between adjacency matrix and adjacency list for sparse graphs.",
    ]

    diff_model = store.get_model("difficulty")
    blooms_model = store.get_model("blooms")
    ct_raw = store.get_model("ct")
    ct_pipe = ct_raw["pipeline"] if isinstance(ct_raw, dict) else ct_raw

    ct_labels = [
        "Decomposition",
        "Pattern Recognition",
        "Abstraction",
        "Algorithmic Thinking",
        "Evaluation & Debugging",
    ]

    for i, q in enumerate(test_questions, 1):
        diff = diff_model.predict([q])[0]
        blooms = blooms_model.predict([q])[0]
        ct_pred = ct_pipe.predict([q])[0]
        ct_active = [ct_labels[idx] for idx, val in enumerate(ct_pred) if val == 1] or [
            "(none detected)"
        ]

        print(f'\n  [Question {i}]: "{q}"')
        print(f"    * Predicted Difficulty : \033[1;32m{diff}\033[0m")
        print(f"    * Bloom's Taxonomy     : \033[1;34m{blooms}\033[0m")
        print(f"    * CT Skills Detected   : \033[1;35m{', '.join(ct_active)}\033[0m")


def demo_step_3_artifacts_manifest():
    print_header(
        "Step 3: Complete ML Artifacts Tracking",
        "RVCE AI254TA: 11 Standard ML Artifact Categories (verified)",
    )
    from src.serving.model_loader import ModelStore

    store = ModelStore(models_dir="models")
    store.load_all()
    manifest = store.get_artifact_manifest()

    print(f"\n{'Category':<32} | {'Tracking Description & Path'}")
    print("-" * 75)
    for k, v in manifest.items():
        clean_k = k.replace("_", " ").title()
        print(f"{clean_k:<32} | {v}")


def demo_step_4_deployment_strategies():
    print_header(
        "Step 4: Blue-Green & Canary Deployment Strategies",
        "RVCE AI254TA Unit IV: Model Promotion & Traffic Routing",
    )
    from collections import Counter

    from fastapi.testclient import TestClient

    import src.serving.app as serving
    from src.registry.model_manager import ModelRegistryManager

    manager = ModelRegistryManager()
    manager.set_demo_state("difficulty", production=PRODUCTION_VERSION, staging=STAGING_VERSION)

    def show_info(api, label):
        info = api.get("/models/info").json()
        print(f"    GET /models/info  [{label}]")
        print(
            f"      serving   : difficulty={info['models'].get('difficulty')}  canary={info['canary_models'].get('difficulty')}"
            f"  ratio={info['canary_ratio']:.0%}"
        )
        print(f"      registry  : difficulty stages={info['registry_stages'].get('difficulty')}")

    # The real FastAPI app, in-process (same code path as `uvicorn src.serving.app:app`)
    with TestClient(serving.app) as api:
        print(
            f"\n[+] 0. START: v{PRODUCTION_VERSION} (LR) in Production, v{STAGING_VERSION} (XGBoost) in Staging"
        )
        show_info(api, "baseline")

        print("\n[+] 1. CANARY at 10%:  POST /deploy/canary?ratio=0.1")
        api.post("/deploy/canary", params={"ratio": 0.1})
        show_info(api, "canary 10%")
        served = Counter(serving.model_store.route("difficulty")[1] for _ in range(1000))
        print(
            "      1,000 routed requests -> "
            + ", ".join(f"{k}: {v / 10:.1f}%" for k, v in sorted(served.items()))
        )

        print("\n[+] 2. BLUE-GREEN PROMOTION: quality gates -> promote -> POST /models/reload")
        gates = manager.evaluate_promotion("difficulty")
        print(
            f"      gates {gates['passed']}/{gates['total_gates']} passed -> {'APPROVED' if gates['promotion_allowed'] else 'BLOCKED'}"
        )
        if gates["promotion_allowed"]:
            manager.promote_to_production(
                "difficulty", version=STAGING_VERSION, archive_existing=True
            )
            api.post("/deploy/canary", params={"ratio": 0.0})
            api.post("/models/reload")
            show_info(api, "after promotion")

        print("\n[+] 3. ROLLBACK: promote_model.py --rollback -> POST /models/reload")
        t0 = time.perf_counter()
        rb = manager.rollback("difficulty")
        api.post("/models/reload")
        ms = (time.perf_counter() - t0) * 1000
        print(
            f"      rolled back to v{rb.get('rolled_back_to_version')} and reloaded in {ms:.0f} ms (measured)"
        )
        show_info(api, "after rollback")

    manager.set_demo_state("difficulty", production=PRODUCTION_VERSION, staging=STAGING_VERSION)
    print("\n    (Registry reset to baseline so the demo can be re-run)")


def main():
    print("\n" + "#" * 75)
    print("  QUESTIONCRAFT PHASE 2 — OFFICIAL EVALUATION DEMO WALKTHROUGH")
    print("  RV College of Engineering (RVCE) | 5th Sem MLOps (AI254TA)")
    print("#" * 75)

    demo_step_1_quality_gates()
    demo_step_2_live_inference()
    demo_step_3_artifacts_manifest()
    demo_step_4_deployment_strategies()

    print("\n" + "=" * 75)
    print("  [SUCCESS] All Phase 2 Demonstration Modules Executed Successfully!")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
