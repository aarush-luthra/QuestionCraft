"""Score the saved models on the synthetic held-out bank (and print per-task metrics).

NOTE: the holdout is LLM-authored in the same style as the synthetic train bank, so these scores
are optimistic. The honest numbers are the rvce_* metrics logged to MLflow.

Usage: python scripts/eval_synthetic.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.ct_classifier import CT_LABELS  # noqa: E402


def main() -> None:
    qs = json.loads((ROOT / "data/raw/synthetic/synthetic_holdout.json").read_text())["questions"]
    texts = [q["question"] for q in qs]
    print(f"synthetic holdout: {len(qs)} questions\n")

    for name, key, labels in (
        ("difficulty", "difficulty", ["Easy", "Medium", "Hard"]),
        ("blooms", "blooms_level", [f"L{i}" for i in range(1, 7)]),
    ):
        pipe = joblib.load(ROOT / f"models/{name}_classifier.pkl")
        y = [q[key] for q in qs]
        pred = pipe.predict(texts)
        maj = max(set(y), key=y.count)
        print(
            f"{name:10s} acc={accuracy_score(y, pred):.3f} "
            f"f1_macro={f1_score(y, pred, labels=labels, average='macro', zero_division=0):.3f} "
            f"(majority baseline acc={y.count(maj) / len(y):.3f})"
        )

    ct = joblib.load(ROOT / "models/ct_classifier.pkl")
    y = ct["mlb"].transform([q["ct_skills"] for q in qs])
    pred = (ct["pipeline"].predict_proba(texts) >= ct["threshold"]).astype(int)
    print(
        f"{'ct':10s} f1_micro={f1_score(y, pred, average='micro', zero_division=0):.3f} "
        f"f1_macro={f1_score(y, pred, average='macro', zero_division=0):.3f} "
        f"(labels: {', '.join(CT_LABELS)})"
    )
    assert np.asarray(y).shape == pred.shape


if __name__ == "__main__":
    main()
