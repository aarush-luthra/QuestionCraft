"""
src/models/duplicate_detector.py
==================================
Semantic duplicate detection using sentence-transformers (all-MiniLM-L6-v2)
and cosine similarity.

Flags question pairs whose cosine similarity exceeds a configurable threshold.
Results are logged as an MLflow artifact.

Usage:
    python -m src.models.duplicate_detector
    python -m src.models.duplicate_detector --input data/raw/rvce/question_bank.json
    python -m src.models.duplicate_detector --input-parquet data/processed/difficulty_dataset.parquet
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from itertools import combinations

import click
import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("questioncraft.duplicate_detector")


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Loader helpers ─────────────────────────────────────────────────────────────

def load_from_json(json_path: Path) -> pd.DataFrame:
    with open(json_path) as f:
        bank = json.load(f)
    rows = [
        {"id": q.get("id", f"q_{i}"), "question_text": q["question"]}
        for i, q in enumerate(bank.get("questions", []))
    ]
    return pd.DataFrame(rows)


def load_from_parquet(parquet_path: Path, sample_n: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df = df.dropna(subset=["question_text"])
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=SEED)
    if "id" not in df.columns:
        df["id"] = [f"row_{i}" for i in range(len(df))]
    return df[["id", "question_text"]].reset_index(drop=True)


# ── Core detection logic ───────────────────────────────────────────────────────

class DuplicateDetector:
    """
    Detects near-duplicate questions using semantic embeddings.

    Parameters
    ----------
    model_name : str
        SentenceTransformer model identifier (default: all-MiniLM-L6-v2).
    threshold : float
        Cosine similarity threshold above which a pair is flagged as duplicate.
    batch_size : int
        Encoding batch size.
    device : str
        "cpu" or "cuda".
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.85,
        batch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self.device = device
        logger.info("Loading SentenceTransformer: %s on %s …", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into L2-normalised embeddings."""
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def find_duplicates(
        self, df: pd.DataFrame, text_col: str = "question_text", id_col: str = "id"
    ) -> pd.DataFrame:
        """
        Compute pairwise cosine similarity and return a DataFrame of
        flagged pairs above the threshold.

        Returns
        -------
        pd.DataFrame
            Columns: id_a, question_a, id_b, question_b, cosine_similarity
        """
        texts = df[text_col].astype(str).tolist()
        ids = df[id_col].astype(str).tolist()

        logger.info("Encoding %d questions …", len(texts))
        embeddings = self.encode(texts)

        logger.info("Computing pairwise cosine similarity …")
        sim_matrix = cosine_similarity(embeddings)

        # Collect upper-triangle pairs above threshold
        flagged = []
        n = len(texts)
        for i, j in combinations(range(n), 2):
            sim = float(sim_matrix[i, j])
            if sim >= self.threshold:
                flagged.append({
                    "id_a": ids[i],
                    "question_a": texts[i],
                    "id_b": ids[j],
                    "question_b": texts[j],
                    "cosine_similarity": round(sim, 6),
                })

        if not flagged:
            result_df = pd.DataFrame(columns=["id_a", "question_a", "id_b", "question_b", "cosine_similarity"])
        else:
            result_df = pd.DataFrame(flagged).sort_values("cosine_similarity", ascending=False)
        logger.info(
            "Found %d duplicate pair(s) above threshold=%.2f out of %d pairs.",
            len(result_df),
            self.threshold,
            n * (n - 1) // 2,
        )
        return result_df

    def get_embeddings_df(self, df: pd.DataFrame, text_col: str = "question_text") -> pd.DataFrame:
        """Return a DataFrame of embeddings (for downstream use / storage)."""
        texts = df[text_col].astype(str).tolist()
        embeddings = self.encode(texts)
        emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
        return pd.concat([df.reset_index(drop=True), emb_df], axis=1)


# ── Main CLI ───────────────────────────────────────────────────────────────────

@click.command()
@click.option("--config", default="configs/model_config.yaml", show_default=True)
@click.option("--input", "json_input", default=None, help="Path to RVCE question_bank.json")
@click.option("--input-parquet", default=None, help="Path to a processed Parquet file")
@click.option("--sample-n", default=500, show_default=True, help="Max questions to sample from Parquet")
@click.option("--out-dir", default="data/processed", show_default=True)
def main(
    config: str,
    json_input: str | None,
    input_parquet: str | None,
    sample_n: int,
    out_dir: str,
) -> None:
    cfg = load_config(config)
    dd_cfg = cfg["duplicate_detector"]

    # ── Determine source ───────────────────────────────────────────────────────
    if json_input:
        df = load_from_json(Path(json_input))
        source_label = Path(json_input).stem
    elif input_parquet:
        df = load_from_parquet(Path(input_parquet), sample_n=sample_n)
        source_label = Path(input_parquet).stem
    else:
        # Default: RVCE question bank
        default_path = Path(cfg["data"]["rvce_bank"])
        logger.info("No input specified; defaulting to %s", default_path)
        df = load_from_json(default_path)
        source_label = "rvce_question_bank"

    logger.info("Input: %d questions from '%s'", len(df), source_label)

    detector = DuplicateDetector(
        model_name=dd_cfg["model_name"],
        threshold=dd_cfg["cosine_threshold"],
        batch_size=dd_cfg["batch_size"],
        device=dd_cfg["device"],
    )

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["global"]["mlflow_tracking_uri"])
    mlflow.set_experiment("questioncraft_duplicate_detector")

    with mlflow.start_run(run_name=f"duplicate_detect_{source_label}") as run:
        logger.info("MLflow run: %s", run.info.run_id)
        mlflow.log_param("model_name", dd_cfg["model_name"])
        mlflow.log_param("threshold", dd_cfg["cosine_threshold"])
        mlflow.log_param("batch_size", dd_cfg["batch_size"])
        mlflow.log_param("device", dd_cfg["device"])
        mlflow.log_param("num_questions", len(df))
        mlflow.log_param("source", source_label)

        # Detect duplicates
        dup_df = detector.find_duplicates(df)

        # Compute & log stats
        total_pairs = len(df) * (len(df) - 1) // 2
        dup_rate = len(dup_df) / total_pairs if total_pairs > 0 else 0.0
        mlflow.log_metrics({
            "num_duplicate_pairs": len(dup_df),
            "total_pairs_checked": total_pairs,
            "duplicate_rate": dup_rate,
        })
        if not dup_df.empty:
            mlflow.log_metric("max_similarity", float(dup_df["cosine_similarity"].max()))
            mlflow.log_metric("mean_duplicate_similarity", float(dup_df["cosine_similarity"].mean()))

        # Save results
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        dup_out = out_path / f"duplicates_{source_label}.parquet"
        dup_df.to_parquet(dup_out, index=False)
        mlflow.log_artifact(str(dup_out), artifact_path="duplicates")

        # Also save embeddings for RVCE bank (small enough)
        if len(df) <= 200:
            emb_df = detector.get_embeddings_df(df)
            emb_out = out_path / f"embeddings_{source_label}.parquet"
            emb_df.to_parquet(emb_out, index=False)
            mlflow.log_artifact(str(emb_out), artifact_path="embeddings")
            logger.info("Embeddings saved → %s", emb_out)

        logger.info("Duplicate report saved → %s", dup_out)
        if dup_df.empty:
            logger.info("✅ No duplicates found above threshold %.2f.", dd_cfg["cosine_threshold"])
        else:
            logger.warning("⚠️  %d duplicate pair(s) found!", len(dup_df))
            print(dup_df[["id_a", "id_b", "cosine_similarity"]].to_string(index=False))

    logger.info("Duplicate detection complete.")


if __name__ == "__main__":
    main()
