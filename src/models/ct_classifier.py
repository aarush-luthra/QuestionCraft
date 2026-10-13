"""
src/models/ct_classifier.py
============================
Multi-label classification for Computational Thinking (CT) skills using
TF-IDF + One-vs-Rest Logistic Regression.

CT Labels:
    - Decomposition
    - Pattern Recognition
    - Abstraction
    - Algorithmic Thinking
    - Evaluation & Debugging

All hyperparameters, data hashes, and metrics are logged to a local MLflow experiment.

Usage:
    python -m src.models.ct_classifier
    python -m src.models.ct_classifier --config configs/model_config.yaml
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
from pathlib import Path

import click
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("questioncraft.ct_classifier")

CT_LABELS = [
    "Decomposition",
    "Pattern Recognition",
    "Abstraction",
    "Algorithmic Thinking",
    "Evaluation & Debugging",
]


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def df_md5(df: pd.DataFrame) -> str:
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


# ── Label parsing ──────────────────────────────────────────────────────────────

def parse_ct_labels(raw: str) -> list[str]:
    """Parse pipe-separated CT skill strings back to a list."""
    if pd.isna(raw) or raw == "":
        return []
    return [s.strip() for s in str(raw).split("|") if s.strip() in CT_LABELS]


# ── Model builder ──────────────────────────────────────────────────────────────

def build_pipeline(cfg: dict) -> Pipeline:
    tfidf_cfg = cfg["tfidf"]
    lr_cfg = cfg["ct_classifier"]["logistic_regression"]

    tfidf = TfidfVectorizer(
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        sublinear_tf=tfidf_cfg["sublinear_tf"],
        min_df=1,        # CT dataset is small; relax min_df
        max_df=1.0,
        analyzer=tfidf_cfg["analyzer"],
        strip_accents=tfidf_cfg["strip_accents"],
    )
    base_lr = LogisticRegression(
        C=lr_cfg["C"],
        solver=lr_cfg["solver"],
        max_iter=lr_cfg["max_iter"],
        class_weight=lr_cfg["class_weight"],
        random_state=lr_cfg["random_state"],
    )
    ovr = OneVsRestClassifier(base_lr, n_jobs=-1)
    return Pipeline([("tfidf", tfidf), ("clf", ovr)])


# ── Per-label bar chart ────────────────────────────────────────────────────────

def plot_per_label_f1(report: dict, labels: list[str], out_path: Path) -> None:
    f1_scores = [report.get(label, {}).get("f1-score", 0.0) for label in labels]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, f1_scores, color="#6c63ff")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1-Score")
    ax.set_title("CT Skill Classifier — Per-Label F1")
    for bar, score in zip(bars, f1_scores):
        ax.text(score + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Per-label F1 chart saved → %s", out_path)


# ── Main training routine ──────────────────────────────────────────────────────

@click.command()
@click.option("--config", default="configs/model_config.yaml", show_default=True)
@click.option("--data-path", default=None, help="Override processed data path")
@click.option("--model-out", default="models/ct_classifier.pkl", show_default=True)
@click.option(
    "--threshold",
    default=None,
    type=float,
    help="Probability threshold for positive label (overrides config)",
)
def main(config: str, data_path: str | None, model_out: str, threshold: float | None) -> None:
    cfg = load_config(config)
    ct_cfg = cfg["ct_classifier"]
    processed_dir = Path(cfg["data"]["processed_dir"])
    prob_threshold = threshold or ct_cfg.get("similarity_threshold", 0.5)

    # ── Load data ──────────────────────────────────────────────────────────────
    data_file = Path(data_path) if data_path else processed_dir / "ct_dataset.parquet"
    if not data_file.exists():
        logger.error("Data file not found: %s. Run preprocess first.", data_file)
        raise SystemExit(1)

    df = pd.read_parquet(data_file)
    logger.info("Loaded %d samples from %s", len(df), data_file)

    df["ct_skills_list"] = df["ct_skills"].apply(parse_ct_labels)
    df = df[df["ct_skills_list"].apply(len) > 0].dropna(subset=["question_text"])
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    data_hash = df_md5(df)

    logger.info("After filtering: %d samples | data hash: %s", len(df), data_hash)

    # Binarise multi-labels
    mlb = MultiLabelBinarizer(classes=CT_LABELS)
    Y = mlb.fit_transform(df["ct_skills_list"])
    X = df["question_text"].astype(str).tolist()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=cfg["global"]["test_size"],
        random_state=SEED,
    )
    logger.info("Train: %d | Test: %d | Labels: %s", len(X_train), len(X_test), CT_LABELS)

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["global"]["mlflow_tracking_uri"])
    mlflow.set_experiment(ct_cfg["mlflow_experiment"])

    with mlflow.start_run(run_name="tfidf_ovr_lr_ct_baseline") as run:
        logger.info("MLflow run: %s", run.info.run_id)

        mlflow.log_param("data_hash", data_hash)
        mlflow.log_param("data_file", str(data_file))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size_n", len(X_test))
        mlflow.log_param("seed", SEED)
        mlflow.log_param("num_labels", len(CT_LABELS))
        mlflow.log_param("prob_threshold", prob_threshold)
        mlflow.log_param("labels", CT_LABELS)
        mlflow.log_params({f"tfidf_{k}": v for k, v in cfg["tfidf"].items()})
        mlflow.log_params({
            f"lr_{k}": v
            for k, v in ct_cfg["logistic_regression"].items()
        })

        pipeline = build_pipeline(cfg)
        logger.info("Training One-vs-Rest pipeline …")
        pipeline.fit(X_train, Y_train)

        # Predict with probability threshold
        Y_prob = pipeline.predict_proba(X_test)
        Y_pred = (Y_prob >= prob_threshold).astype(int)

        # Metrics
        h_loss = hamming_loss(Y_test, Y_pred)
        f1_micro = f1_score(Y_test, Y_pred, average="micro", zero_division=0)
        f1_macro = f1_score(Y_test, Y_pred, average="macro", zero_division=0)
        f1_samples = f1_score(Y_test, Y_pred, average="samples", zero_division=0)
        prec_micro = precision_score(Y_test, Y_pred, average="micro", zero_division=0)
        rec_micro  = recall_score(Y_test, Y_pred, average="micro", zero_division=0)

        mlflow.log_metrics({
            "test_hamming_loss": h_loss,
            "test_f1_micro": f1_micro,
            "test_f1_macro": f1_macro,
            "test_f1_samples": f1_samples,
            "test_precision_micro": prec_micro,
            "test_recall_micro": rec_micro,
        })
        logger.info(
            "Test → Hamming Loss: %.4f | F1-micro: %.4f | F1-macro: %.4f",
            h_loss, f1_micro, f1_macro,
        )

        # Per-label report
        report = classification_report(
            Y_test, Y_pred, target_names=CT_LABELS, output_dict=True, zero_division=0
        )
        print(classification_report(Y_test, Y_pred, target_names=CT_LABELS, zero_division=0))

        for label in CT_LABELS:
            if label in report:
                slug = label.lower().replace(" ", "_").replace("&", "and")
                mlflow.log_metrics({
                    f"f1_{slug}": report[label]["f1-score"],
                    f"precision_{slug}": report[label]["precision"],
                    f"recall_{slug}": report[label]["recall"],
                })

        # Plot
        tmp_dir = Path("tmp_artifacts")
        tmp_dir.mkdir(exist_ok=True)
        chart_path = tmp_dir / "ct_per_label_f1.png"
        plot_per_label_f1(report, CT_LABELS, chart_path)
        mlflow.log_artifact(str(chart_path), artifact_path="plots")

        # Save model & MLB
        model_path = Path(model_out)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": pipeline, "mlb": mlb, "threshold": prob_threshold}, model_path)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "questioncraft_ct_classifier")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
