"""
scripts/run_phase1_pipeline.py
==============================
CLI orchestrator to execute Phase 1 pipeline stages:
1. Data Ingestion (download)
2. Preprocessing & Feature Extraction
3. Model Training (Difficulty, Bloom's Taxonomy, Computational Thinking)
4. Duplicate Question Detection

Usage:
    python scripts/run_phase1_pipeline.py --help
    python scripts/run_phase1_pipeline.py --stage all
    python scripts/run_phase1_pipeline.py --stage train
    python scripts/run_phase1_pipeline.py --stage dedup
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_stage(name: str, command: list[str]) -> bool:
    print(f"\n{'='*70}")
    print(f"[*] Running Stage: {name}")
    print(f"[*] Command: {' '.join(command)}")
    print(f"{'='*70}\n")
    start_time = time.time()
    result = subprocess.run(command, check=False)
    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"\n[+] Stage '{name}' completed successfully in {elapsed:.2f}s")
        return True
    else:
        print(f"\n[-] Stage '{name}' failed with returncode {result.returncode} in {elapsed:.2f}s")
        return False


def main():
    parser = argparse.ArgumentParser(description="QuestionCraft Phase 1 Pipeline Runner")
    parser.add_argument(
        "--stage",
        choices=["all", "download", "preprocess", "train", "dedup"],
        default="all",
        help="Pipeline stage to execute (default: all)",
    )
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    py = sys.executable

    stages = {
        "download": [
            ("Download Benchmark Datasets", [py, "-m", "src.data.download", "--config", args.config]),
        ],
        "preprocess": [
            ("Preprocess & Feature Extraction", [py, "-m", "src.data.preprocess", "--config", args.config]),
        ],
        "train": [
            ("Train Difficulty Classifier", [py, "-m", "src.models.difficulty_classifier", "--config", args.config, "--model-out", "models/difficulty_classifier.pkl"]),
            ("Train Bloom's Taxonomy Classifier", [py, "-m", "src.models.blooms_classifier", "--config", args.config, "--model-out", "models/blooms_classifier.pkl"]),
            ("Train CT Classifier", [py, "-m", "src.models.ct_classifier", "--config", args.config, "--model-out", "models/ct_classifier.pkl"]),
        ],
        "dedup": [
            ("Semantic Duplicate Detection", [py, "-m", "src.models.duplicate_detector", "--config", args.config, "--input", "data/raw/rvce/question_bank.json", "--out-dir", "models"]),
        ],
    }

    to_run = []
    if args.stage == "all":
        for s in ["download", "preprocess", "train", "dedup"]:
            to_run.extend(stages[s])
    else:
        to_run.extend(stages[args.stage])

    total_start = time.time()
    success_count = 0

    for name, cmd in to_run:
        ok = run_stage(name, cmd)
        if ok:
            success_count += 1
        else:
            print(f"\n[!] Pipeline halted due to failure in stage: {name}")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f" QuestionCraft Phase 1 Pipeline Complete: {success_count}/{len(to_run)} stages passed in {total_elapsed:.2f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
