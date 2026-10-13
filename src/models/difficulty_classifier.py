"""
src/models/difficulty_classifier.py
=====================================
TF-IDF + Logistic Regression baseline for 3-class difficulty classification.
(Easy / Medium / Hard)

All hyperparameters, data hashes, and metrics are logged to a local MLflow experiment.

Usage:
    python -m src.models.difficulty_classifier
    python -m src.models.difficulty_classifier --config configs/model_config.yaml
"""

from __future__ import annotations

import hashlib
import json
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

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
logger = logging.getLogger("questioncraft.difficulty_classifier")


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def df_md5(df: pd.DataFrame) -> str:
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


# ── Model Builder ──────────────────────────────────────────────────────────────

def build_pipeline(cfg: dict) -> Pipeline:
    tfidf_cfg = cfg["tfidf"]
    lr_cfg = cfg["difficulty_classifier"]["logistic_regression"]

    tfidf = TfidfVectorizer(
        max_features=tfidf_cfg["max_features"],
        ngram_range=tuple(tfidf_cfg["ngram_range"]),
        sublinear_tf=tfidf_cfg["sublinear_tf"],
        min_df=tfidf_cfg["min_df"],
        max_df=tfidf_cfg["max_df"],
        analyzer=tfidf_cfg["analyzer"],
        strip_accents=tfidf_cfg["strip_accents"],
    )
    lr = LogisticRegression(
        C=lr_cfg["C"],
        solver=lr_cfg["solver"],
        max_iter=lr_cfg["max_iter"],
        multi_class=lr_cfg["multi_class"],
        class_weight=lr_cfg["class_weight"],
        random_state=lr_cfg["random_state"],
    )
    return Pipeline([("tfidf", tfidf), ("clf", lr)])


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, labels: list[str], out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Difficulty Classifier — Confusion Matrix")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", out_path)


# ── Main training routine ──────────────────────────────────────────────────────

@click.command()
@click.option("--config", default="configs/model_config.yaml", show_default=True)
@click.option("--data-path", default=None, help="Override processed data path")
@click.option("--model-out", default="models/difficulty_classifier.pkl", show_default=True)
def main(config: str, data_path: str | None, model_out: str) -> None:
    cfg = load_config(config)
    diff_cfg = cfg["difficulty_classifier"]
    processed_dir = Path(cfg["data"]["processed_dir"])
    labels = diff_cfg["labels"]

    # ── Load data ──────────────────────────────────────────────────────────────
    data_file = Path(data_path) if data_path else processed_dir / "difficulty_dataset.parquet"
    if not data_file.exists():
        logger.error("Data file not found: %s. Run preprocess first.", data_file)
        raise SystemExit(1)

    df = pd.read_parquet(data_file)
    logger.info("Loaded %d samples from %s", len(df), data_file)

    df = df[df["difficulty"].isin(labels)].dropna(subset=["question_text", "difficulty"])
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle
    data_hash = df_md5(df)
    logger.info("Data hash: %s | Class distribution:\n%s", data_hash, df["difficulty"].value_counts())

    X = df["question_text"].astype(str).tolist()
    y = df["difficulty"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["global"]["test_size"],
        random_state=SEED,
        stratify=y,
    )
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    # ── MLflow experiment ──────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["global"]["mlflow_tracking_uri"])
    mlflow.set_experiment(diff_cfg["mlflow_experiment"])

    with mlflow.start_run(run_name="tfidf_lr_baseline") as run:
        logger.info("MLflow run: %s", run.info.run_id)

        # Log params
        mlflow.log_param("data_hash", data_hash)
        mlflow.log_param("data_file", str(data_file))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size_n", len(X_test))
        mlflow.log_param("seed", SEED)
        mlflow.log_params({f"tfidf_{k}": v for k, v in cfg["tfidf"].items()})
        mlflow.log_params({
            f"lr_{k}": v
            for k, v in diff_cfg["logistic_regression"].items()
        })

        # Build & train
        pipeline = build_pipeline(cfg)
        logger.info("Training pipeline …")
        pipeline.fit(X_train, y_train)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        mlflow.log_metric("cv_f1_macro_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_f1_macro_std", float(cv_scores.std()))
        logger.info("CV F1-macro: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

        # Test evaluation
        y_pred = pipeline.predict(X_test)
        acc   = accuracy_score(y_test, y_pred)
        f1    = f1_score(y_test, y_pred, average="macro", zero_division=0)
        prec  = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec   = recall_score(y_test, y_pred, average="macro", zero_division=0)

        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_f1_macro": f1,
            "test_precision_macro": prec,
            "test_recall_macro": rec,
        })
        logger.info("Test → Accuracy: %.4f | F1-macro: %.4f", acc, f1)
        print(classification_report(y_test, y_pred, labels=labels, zero_division=0))

        # Per-class metrics
        report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
        for label in labels:
            if label in report:
                mlflow.log_metrics({
                    f"f1_{label.lower()}": report[label]["f1-score"],
                    f"precision_{label.lower()}": report[label]["precision"],
                    f"recall_{label.lower()}": report[label]["recall"],
                })

        # Confusion matrix artifact
        tmp_dir = Path("tmp_artifacts")
        tmp_dir.mkdir(exist_ok=True)
        cm_path = tmp_dir / "difficulty_confusion_matrix.png"
        plot_confusion_matrix(y_test, y_pred, labels, cm_path)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

        # Save model
        model_path = Path(model_out)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        mlflow.log_artifact(str(model_path))
        logger.info("Model saved → %s", model_path)

        # Register model
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "questioncraft_difficulty_classifier")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
