"""
src/data/download.py
====================
Downloads ARC (Easy + Challenge), RACE, and SciQ datasets from HuggingFace
and saves each split as a Parquet file under data/raw/.

Usage:
    python -m src.data.download
    python -m src.data.download --config configs/model_config.yaml
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path

import click
import numpy as np
import yaml
from datasets import load_dataset

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
logger = logging.getLogger("questioncraft.download")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def file_md5(path: Path) -> str:
    """Return MD5 hex-digest of a file (used for DVC-style data hashing)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_manifest(manifest: dict, out_dir: Path) -> None:
    manifest_path = out_dir / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved → %s", manifest_path)


# ── Dataset downloaders ────────────────────────────────────────────────────────

def download_arc(raw_dir: Path, cfg: dict) -> dict[str, str]:
    """Download ARC-Challenge and ARC-Easy, return {split_name: parquet_path}."""
    records: dict[str, str] = {}
    hf_path = cfg.get("data", {}).get("datasets", {}).get("arc", {}).get("hf_path", "allenai/ai2_arc")
    for subset in ("ARC-Challenge", "ARC-Easy"):
        logger.info("Downloading ARC subset: %s …", subset)
        ds = load_dataset(hf_path, subset)
        subset_slug = subset.lower().replace("-", "_")
        out_dir = raw_dir / "arc" / subset_slug
        out_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_ds in ds.items():
            # Normalise columns: keep question + answerKey + choices
            split_ds = split_ds.remove_columns(
                [c for c in split_ds.column_names if c not in ("id", "question", "choices", "answerKey")]
            )
            out_path = out_dir / f"{split_name}.parquet"
            split_ds.to_parquet(str(out_path))
            records[f"arc/{subset_slug}/{split_name}"] = str(out_path)
            logger.info("  Saved %d rows → %s", len(split_ds), out_path)

    return records


def download_race(raw_dir: Path, cfg: dict | None = None) -> dict[str, str]:
    """Download RACE (all), return {split_name: parquet_path}."""
    logger.info("Downloading RACE …")
    hf_path = "ehovy/race"
    if cfg and "data" in cfg and "datasets" in cfg["data"] and "race" in cfg["data"]["datasets"]:
        hf_path = cfg["data"]["datasets"]["race"].get("hf_path", "ehovy/race")
    ds = load_dataset(hf_path, "all")
    out_dir = raw_dir / "race"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, str] = {}

    for split_name, split_ds in ds.items():
        split_ds = split_ds.remove_columns(
            [c for c in split_ds.column_names if c not in ("example_id", "article", "question", "answer", "options")]
        )
        out_path = out_dir / f"{split_name}.parquet"
        split_ds.to_parquet(str(out_path))
        records[f"race/{split_name}"] = str(out_path)
        logger.info("  Saved %d rows → %s", len(split_ds), out_path)

    return records


def download_sciq(raw_dir: Path, cfg: dict | None = None) -> dict[str, str]:
    """Download SciQ, return {split_name: parquet_path}."""
    logger.info("Downloading SciQ …")
    hf_path = "allenai/sciq"
    if cfg and "data" in cfg and "datasets" in cfg["data"] and "sciq" in cfg["data"]["datasets"]:
        hf_path = cfg["data"]["datasets"]["sciq"].get("hf_path", "allenai/sciq")
    ds = load_dataset(hf_path)
    out_dir = raw_dir / "sciq"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, str] = {}

    for split_name, split_ds in ds.items():
        out_path = out_dir / f"{split_name}.parquet"
        split_ds.to_parquet(str(out_path))
        records[f"sciq/{split_name}"] = str(out_path)
        logger.info("  Saved %d rows → %s", len(split_ds), out_path)

    return records


def download_blooms_kaggle(raw_dir: Path) -> dict[str, str]:
    """
    Attempt to load a Bloom's taxonomy dataset from HuggingFace Hub.
    Falls back gracefully if the dataset is unavailable.
    """
    records: dict[str, str] = {}
    candidate_datasets = [
        "itsskofficial/falcon-7b-blooms-taxonomy-data",
        "LLukas22/bloom-taxonomy-dataset",
    ]
    for repo_id in candidate_datasets:
        try:
            logger.info("Downloading Bloom's taxonomy dataset from %s …", repo_id)
            ds = load_dataset(repo_id)
            out_dir = raw_dir / "blooms"
            out_dir.mkdir(parents=True, exist_ok=True)

            for split_name, split_ds in ds.items():
                out_path = out_dir / f"{split_name}.parquet"
                split_ds.to_parquet(str(out_path))
                records[f"blooms/{split_name}"] = str(out_path)
                logger.info("  Saved %d rows → %s", len(split_ds), out_path)
            if records:
                break
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dataset %s unavailable: %s. Trying next...", repo_id, exc)

    return records


# ── Entry point ────────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--config",
    default="configs/model_config.yaml",
    show_default=True,
    help="Path to model_config.yaml",
)
@click.option(
    "--raw-dir",
    default=None,
    help="Override raw data directory from config",
)
def main(config: str, raw_dir: str | None) -> None:
    cfg = load_config(config)
    raw_path = Path(raw_dir or cfg["data"]["raw_dir"])
    raw_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("QuestionCraft — Data Download")
    logger.info("Raw directory: %s", raw_path.resolve())
    logger.info("=" * 60)

    all_records: dict[str, str] = {}
    all_records.update(download_arc(raw_path, cfg))
    all_records.update(download_race(raw_path, cfg))
    all_records.update(download_sciq(raw_path, cfg))
    all_records.update(download_blooms_kaggle(raw_path))

    # Build manifest with file hashes
    manifest = {
        "seed": SEED,
        "files": {
            key: {"path": path, "md5": file_md5(Path(path))}
            for key, path in all_records.items()
        },
    }
    save_manifest(manifest, raw_path)
    logger.info("Download complete. %d file(s) saved.", len(all_records))


if __name__ == "__main__":
    main()
