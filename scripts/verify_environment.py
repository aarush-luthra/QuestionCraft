"""
scripts/verify_environment.py
=============================
Diagnostic script to verify the local runtime environment for QuestionCraft Phase 1:
- Python version (>= 3.10)
- Essential package installations (scikit-learn, XGBoost, PyTorch, Transformers, MLflow, DVC)
- NLTK corpora bootstrap status
- Local configuration and dataset directory structure

Usage:
    python scripts/verify_environment.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REQUIRED_PACKAGES = [
    "sklearn",
    "xgboost",
    "pandas",
    "numpy",
    "scipy",
    "transformers",
    "sentence_transformers",
    "torch",
    "nltk",
    "textstat",
    "datasets",
    "mlflow",
    "dvc",
    "yaml",
]


def check_python_version() -> bool:
    print(f"[*] Python Version: {sys.version.split()[0]}", end=" ")
    if sys.version_info >= (3, 10):
        print("[OK]")
        return True
    print("[FAIL: Python 3.10+ required]")
    return False


def check_packages() -> bool:
    print("\n[*] Checking core dependencies:")
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "installed")
            print(f"  - {pkg:<24} : {version} [OK]")
        except ImportError:
            print(f"  - {pkg:<24} : NOT FOUND [MISSING]")
            all_ok = False
    return all_ok


def check_nltk_resources() -> bool:
    print("\n[*] Checking NLTK resources:")
    import nltk
    resources = [
        "tokenizers/punkt",
        "taggers/averaged_perceptron_tagger",
        "taggers/averaged_perceptron_tagger_eng",
        "corpora/stopwords",
    ]
    all_ok = True
    for res in resources:
        try:
            nltk.data.find(res)
            print(f"  - {res:<35} : [OK]")
        except LookupError:
            print(f"  - {res:<35} : [DOWNLOADING]")
            nltk.download(res.split("/")[-1], quiet=True)
    return all_ok


def check_directory_structure() -> bool:
    print("\n[*] Checking project directory structure:")
    repo_root = Path(__file__).resolve().parent.parent
    paths = [
        ("configs/model_config.yaml", True),
        ("src/data/feature_engineering.py", True),
        ("src/models/difficulty_classifier.py", True),
        ("src/models/blooms_classifier.py", True),
        ("src/models/ct_classifier.py", True),
        ("src/models/duplicate_detector.py", True),
        ("data", False),
        ("models", False),
        ("reports", False),
        ("notebooks", False),
    ]
    all_ok = True
    for rel_path, is_file in paths:
        target = repo_root / rel_path
        status = target.is_file() if is_file else target.is_dir()
        icon = "[OK]" if status else "[MISSING]"
        print(f"  - {rel_path:<35} : {icon}")
        if not status:
            all_ok = False
    return all_ok


def main():
    print("=" * 60)
    print(" QuestionCraft Phase 1: Environment & Dependency Verification")
    print("=" * 60)

    py_ok = check_python_version()
    pkg_ok = check_packages()
    nltk_ok = check_nltk_resources()
    dir_ok = check_directory_structure()

    print("\n" + "=" * 60)
    if py_ok and pkg_ok and nltk_ok and dir_ok:
        print(" Verification SUCCESS: Environment is ready for Phase 1.")
    else:
        print(" Verification WARNING: Some components are missing or unconfigured.")
    print("=" * 60)


if __name__ == "__main__":
    main()
