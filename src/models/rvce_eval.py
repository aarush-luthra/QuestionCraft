"""
src/models/rvce_eval.py
=======================
Held-out evaluation on real RVCE exam questions (built by src/data/build_splits.py).

The models train on public / proxy data; these helpers score them on RVCE papers they have
never seen and return MLflow-ready metrics prefixed with ``rvce_``. Labels in the eval set come
from printed paper metadata (Bloom's) or from LLM drafts awaiting faculty review (difficulty,
CT), so the label-source mix is returned too and should be reported alongside the numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss


def load_eval(eval_path: str | None) -> pd.DataFrame | None:
    if not eval_path or not Path(eval_path).exists():
        return None
    df = pd.read_parquet(eval_path)
    return df if not df.empty else None


def evaluate_single_label(pipeline, eval_path: str | None, labels: list[str]) -> dict:
    """Accuracy / macro-F1 of a single-label classifier on the RVCE held-out set."""
    df = load_eval(eval_path)
    if df is None:
        return {}
    df = df[df["label"].isin(labels)]
    y_pred = pipeline.predict(df["question_text"].astype(str).tolist())
    metrics = {
        "rvce_accuracy": float(accuracy_score(df["label"], y_pred)),
        "rvce_f1_macro": float(
            f1_score(df["label"], y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "rvce_n": float(len(df)),
    }
    # Majority-class baseline, so the number can be judged honestly
    majority = df["label"].value_counts().idxmax()
    metrics["rvce_majority_baseline_accuracy"] = float((df["label"] == majority).mean())
    return metrics


def evaluate_multilabel(pipeline, mlb, threshold: float, eval_path: str | None, parse) -> dict:
    """Micro/macro-F1 of the multi-label CT classifier on the RVCE held-out set."""
    df = load_eval(eval_path)
    if df is None:
        return {}
    y_true = mlb.transform(df["label"].apply(parse))
    y_pred = (pipeline.predict_proba(df["question_text"].astype(str).tolist()) >= threshold).astype(
        int
    )
    return {
        "rvce_f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "rvce_f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "rvce_hamming_loss": float(hamming_loss(y_true, y_pred)),
        "rvce_n": float(len(df)),
        "rvce_label_cardinality": float(np.asarray(y_true).sum(axis=1).mean()),
    }


def label_source_mix(eval_path: str | None) -> str:
    df = load_eval(eval_path)
    if df is None or "label_source" not in df:
        return ""
    return ",".join(f"{k}:{v}" for k, v in df["label_source"].value_counts().items())
