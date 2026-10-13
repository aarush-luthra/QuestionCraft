"""
src/data/preprocess.py
======================
Reads raw Parquet files, applies FeatureExtractor, and writes
processed Parquet files ready for model training.

DVC stage: preprocess
  deps : data/raw/
  outs : data/processed/

Usage:
    python -m src.data.preprocess
    python -m src.data.preprocess --config configs/model_config.yaml
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
import pandas as pd
import yaml

from src.data.feature_engineering import FeatureExtractor

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
logger = logging.getLogger("questioncraft.preprocess")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def df_md5(df: pd.DataFrame) -> str:
    """Deterministic hash of a DataFrame for MLflow traceability."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


# ── Dataset-specific loaders → normalise to (question, label, *metadata) ──────

def load_arc(raw_dir: Path) -> pd.DataFrame:
    """Load ARC-Challenge + ARC-Easy and assign difficulty labels."""
    frames = []
    for subset in ("arc_challenge", "arc_easy"):
        difficulty = "Hard" if subset == "arc_challenge" else "Easy"
        subset_dir = raw_dir / "arc" / subset
        if not subset_dir.exists():
            logger.warning("ARC subset dir not found: %s", subset_dir)
            continue
        for parquet in subset_dir.glob("*.parquet"):
            df = pd.read_parquet(parquet)
            df = df.rename(columns={"question": "question_text"})
            df["difficulty"] = difficulty
            df["source"] = f"arc_{subset}"
            frames.append(df[["question_text", "difficulty", "source"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_race(raw_dir: Path) -> pd.DataFrame:
    """Load RACE; map article-level difficulty (middle=Easy, high=Hard)."""
    frames = []
    race_dir = raw_dir / "race"
    if not race_dir.exists():
        logger.warning("RACE dir not found: %s", race_dir)
        return pd.DataFrame()
    for parquet in race_dir.glob("*.parquet"):
        df = pd.read_parquet(parquet)
        df = df.rename(columns={"question": "question_text"})
        df["source"] = "race"
        frames.append(df[["question_text", "source"]])
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result["difficulty"] = "Medium"   # RACE questions are generally medium difficulty
    return result


def load_sciq(raw_dir: Path) -> pd.DataFrame:
    """Load SciQ dataset."""
    frames = []
    sciq_dir = raw_dir / "sciq"
    if not sciq_dir.exists():
        logger.warning("SciQ dir not found: %s", sciq_dir)
        return pd.DataFrame()
    for parquet in sciq_dir.glob("*.parquet"):
        df = pd.read_parquet(parquet)
        df = df.rename(columns={"question": "question_text"})
        df["source"] = "sciq"
        df["difficulty"] = "Medium"
        frames.append(df[["question_text", "difficulty", "source"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_blooms(raw_dir: Path) -> pd.DataFrame:
    """Load Bloom's taxonomy dataset from Parquet and CSV files."""
    blooms_dir = raw_dir / "blooms"
    if not blooms_dir.exists():
        logger.warning("Bloom's dir not found: %s", blooms_dir)
        return pd.DataFrame()
    frames = []
    # 1. Parquet files
    for parquet in blooms_dir.glob("*.parquet"):
        df = pd.read_parquet(parquet)
        col_map = {}
        for col in df.columns:
            if "question" in col.lower():
                col_map[col] = "question_text"
            if "bloom" in col.lower() or "level" in col.lower():
                col_map[col] = "blooms_level"
        df = df.rename(columns=col_map)
        df["source"] = f"blooms_parquet_{parquet.stem}"
        if "question_text" in df.columns and "blooms_level" in df.columns:
            frames.append(df[["question_text", "blooms_level", "source"]])

    # 2. CSV files (e.g. questions.csv from bloomsdata.zip)
    for csv_file in blooms_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        col_map = {}
        for col in df.columns:
            if col.lower() in ("question", "question_text"):
                col_map[col] = "question_text"
            if "bloom" in col.lower() or "level" in col.lower():
                col_map[col] = "blooms_level"
        df = df.rename(columns=col_map)
        df["source"] = f"blooms_csv_{csv_file.stem}"
        if "question_text" in df.columns and "blooms_level" in df.columns:
            frames.append(df[["question_text", "blooms_level", "source"]])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_blooms_difficulty(raw_dir: Path) -> pd.DataFrame:
    """Extract difficulty annotations from Bloom's CSV files if present."""
    blooms_dir = raw_dir / "blooms"
    if not blooms_dir.exists():
        return pd.DataFrame()
    frames = []
    for csv_file in blooms_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        col_map = {}
        for col in df.columns:
            if col.lower() in ("question", "question_text"):
                col_map[col] = "question_text"
            if col.lower() == "difficulty":
                col_map[col] = "difficulty"
        df = df.rename(columns=col_map)
        if "question_text" in df.columns and "difficulty" in df.columns:
            sub = df[["question_text", "difficulty"]].dropna().copy()
            sub["source"] = f"blooms_csv_{csv_file.stem}"
            frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_rvce(bank_path: Path) -> pd.DataFrame:
    """Load a single RVCE question bank JSON.

    Handles two schema variants:
    - Original DSA mock bank: keys = blooms_level, ct_skills (list), course_outcome, subject
    - UHV bank (HS248AT): keys = btl (Bloom's level), ct_skill (str), co, course_code
    """
    if not bank_path.exists():
        logger.warning("RVCE question bank not found: %s", bank_path)
        return pd.DataFrame()
    with open(bank_path) as f:
        bank = json.load(f)
    rows = []
    for q in bank.get("questions", []):
        # Bloom's level — support both 'blooms_level' and 'btl'
        blooms = q.get("blooms_level") or q.get("btl")

        # CT skills — support both list ('ct_skills') and single string ('ct_skill')
        ct_list = q.get("ct_skills")
        if ct_list is None:
            raw_ct = q.get("ct_skill")
            ct_list = [raw_ct] if raw_ct else []
        ct_str = "|".join([c for c in ct_list if c]) if ct_list else ""

        # Course outcome — support both 'course_outcome' and 'co'
        co = q.get("course_outcome") or q.get("co")

        # Subject — support 'subject', fall back to 'course_code'
        subject = q.get("subject") or q.get("course_code")

        rows.append({
            "question_id": q.get("id"),
            "question_text": q["question"],
            "difficulty": q.get("difficulty"),
            "blooms_level": blooms,
            "ct_skills": ct_str if ct_str else None,
            "marks": q.get("marks"),
            "course_outcome": co,
            "subject": subject,
            "source": f"rvce_{bank_path.stem}",
        })
    df = pd.DataFrame(rows)
    logger.info("Loaded %d questions from %s", len(df), bank_path.name)
    return df


def load_all_rvce_banks(cfg: dict) -> pd.DataFrame:
    """Load and merge all RVCE question banks listed in config."""
    bank_paths = cfg["data"].get("rvce_banks", [cfg["data"].get("rvce_bank")])
    frames = [load_rvce(Path(p)) for p in bank_paths if p]
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame()
    merged = pd.concat(non_empty, ignore_index=True)
    # Drop full duplicates (same question text across banks)
    merged = merged.drop_duplicates(subset=["question_text"]).reset_index(drop=True)
    logger.info("Total RVCE questions after merge & dedup: %d", len(merged))
    return merged


def extract_features_with_cache(df: pd.DataFrame, fe: FeatureExtractor) -> pd.DataFrame:
    """Extract features for unique questions and merge back, avoiding duplicate compute."""
    unique_qs = df["question_text"].drop_duplicates().tolist()
    logger.info("Extracting features for %d unique questions (out of %d total rows) …", len(unique_qs), len(df))
    feat_df = fe.extract_batch(unique_qs)
    feat_df["question_text"] = unique_qs
    return df.merge(feat_df, on="question_text", how="left")


# ── Main preprocessing pipeline ────────────────────────────────────────────────

@click.command()
@click.option("--config", default="configs/model_config.yaml", show_default=True)
def main(config: str) -> None:
    cfg = load_config(config)
    raw_dir = Path(cfg["data"]["raw_dir"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    fe = FeatureExtractor(config_path=config)
    manifest: dict[str, str] = {}

    # Load all RVCE banks (DSA mock + UHV + any future additions)
    rvce_df = load_all_rvce_banks(cfg)

    # ── 1. Difficulty dataset (ARC + RACE + SciQ + Blooms CSV + RVCE) ────────
    logger.info("Building difficulty dataset …")
    difficulty_frames = [
        load_arc(raw_dir),
        load_race(raw_dir),
        load_sciq(raw_dir),
        load_blooms_difficulty(raw_dir),
    ]
    if not rvce_df.empty and "difficulty" in rvce_df.columns:
        difficulty_frames.append(
            rvce_df[["question_text", "difficulty", "source"]].dropna(subset=["difficulty"])
        )

    difficulty_df = pd.concat(
        [f for f in difficulty_frames if not f.empty], ignore_index=True
    ).dropna(subset=["question_text", "difficulty"])
    difficulty_df = difficulty_df[difficulty_df["difficulty"].isin(["Easy", "Medium", "Hard"])]

    difficulty_out = extract_features_with_cache(difficulty_df, fe)
    out_path = processed_dir / "difficulty_dataset.parquet"
    difficulty_out.to_parquet(out_path, index=False)
    manifest["difficulty"] = df_md5(difficulty_out)
    logger.info("  → %s (%d rows)", out_path, len(difficulty_out))

    # ── 2. Bloom's dataset (Bloom's Parquet + Bloom's CSV + RVCE) ────────────
    logger.info("Building Bloom's dataset …")
    blooms_frames = [load_blooms(raw_dir)]
    if not rvce_df.empty and "blooms_level" in rvce_df.columns:
        blooms_frames.append(
            rvce_df[["question_text", "blooms_level", "source"]].dropna(subset=["blooms_level"])
        )

    blooms_df = pd.concat(
        [f for f in blooms_frames if not f.empty], ignore_index=True
    ).dropna(subset=["question_text", "blooms_level"])

    if not blooms_df.empty:
        blooms_out = extract_features_with_cache(blooms_df, fe)
        out_path = processed_dir / "blooms_dataset.parquet"
        blooms_out.to_parquet(out_path, index=False)
        manifest["blooms"] = df_md5(blooms_out)
        logger.info("  → %s (%d rows)", out_path, len(blooms_out))

    # ── 3. CT dataset (all RVCE banks with CT annotations) ───────────────────
    logger.info("Building CT dataset …")
    if not rvce_df.empty and "ct_skills" in rvce_df.columns:
        ct_df = rvce_df[["question_text", "ct_skills", "source"]].dropna(subset=["ct_skills"])
        ct_df = ct_df[ct_df["ct_skills"].str.strip() != ""]
        if not ct_df.empty:
            ct_out = extract_features_with_cache(ct_df, fe)
            out_path = processed_dir / "ct_dataset.parquet"
            ct_out.to_parquet(out_path, index=False)
            manifest["ct"] = df_md5(ct_out)
            logger.info("  → %s (%d rows)", out_path, len(ct_out))

    # ── Save data hashes manifest ──────────────────────────────────────────────
    manifest_path = processed_dir / "data_hashes.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Data hashes saved → %s", manifest_path)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()

