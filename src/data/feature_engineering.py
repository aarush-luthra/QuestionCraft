"""
src/data/feature_engineering.py
================================
FeatureExtractor class that computes:
  - Readability metrics (Flesch, Coleman-Liau, ARI, Dale-Chall)
  - Structural / surface features (word count, sentence count, POS tags)
  - Bloom's taxonomy verb presence indicators (L1–L6)
  - Computational Thinking keyword indicators

Usage:
    from src.data.feature_engineering import FeatureExtractor
    fe = FeatureExtractor()
    features = fe.extract("Define a stack and list its primary operations.")
"""

from __future__ import annotations

import logging
import os
import re
import string
from pathlib import Path
from typing import Any

import nltk
import numpy as np
import pandas as pd
import textstat
import yaml
from nltk.tokenize import sent_tokenize, word_tokenize

# ── NLTK Resource Bootstrap ────────────────────────────────────────────────────
_NLTK_RESOURCES = [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "stopwords",
]

def _ensure_nltk_resources() -> None:
    for resource in _NLTK_RESOURCES:
        try:
            if "punkt" in resource:
                nltk.data.find(f"tokenizers/{resource}")
            elif "stopwords" in resource:
                nltk.data.find(f"corpora/{resource}")
            else:
                nltk.data.find(f"taggers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

_ensure_nltk_resources()

logger = logging.getLogger("questioncraft.feature_engineering")

# ── Constants ──────────────────────────────────────────────────────────────────
_DEFAULT_CONFIG = "configs/model_config.yaml"


# ── FeatureExtractor ───────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Extracts a fixed-length numerical feature vector from a raw question string.

    Parameters
    ----------
    config_path : str
        Path to model_config.yaml. Falls back to built-in defaults if absent.
    """

    def __init__(self, config_path: str = _DEFAULT_CONFIG) -> None:
        self.cfg = self._load_config(config_path)
        self.fe_cfg = self.cfg.get("feature_engineering", {})

        # Build lookup sets / dicts from config
        self.blooms_verbs: dict[str, set[str]] = {
            level: set(verbs)
            for level, verbs in self.fe_cfg.get("blooms_verbs", {}).items()
        }
        self.ct_keywords: dict[str, set[str]] = {
            skill: set(kws)
            for skill, kws in self.fe_cfg.get("ct_keywords", {}).items()
        }
        self.pos_tags_of_interest: list[str] = self.fe_cfg.get(
            "pos_tags", ["NN", "VB", "JJ", "RB", "WP"]
        )

        logger.info(
            "FeatureExtractor initialised | Bloom levels: %d | CT skills: %d",
            len(self.blooms_verbs),
            len(self.ct_keywords),
        )

    # ── Config loader ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_config(path: str) -> dict:
        if Path(path).exists():
            with open(path) as f:
                return yaml.safe_load(f)
        logger.warning("Config not found at %s; using empty defaults.", path)
        return {}

    # ── Low-level helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        """Light cleaning: strip extra whitespace, lower-case for lookups."""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _tokenize_words(text: str) -> list[str]:
        return word_tokenize(text)

    @staticmethod
    def _tokenize_sentences(text: str) -> list[str]:
        return sent_tokenize(text)

    def _pos_tag(self, tokens: list[str]) -> list[tuple[str, str]]:
        return nltk.pos_tag(tokens)

    # ── Readability features ───────────────────────────────────────────────────

    def _readability_features(self, text: str) -> dict[str, float]:
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "coleman_liau_index": textstat.coleman_liau_index(text),
            "automated_readability_index": textstat.automated_readability_index(text),
            "dale_chall_readability_score": textstat.dale_chall_readability_score(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "smog_index": textstat.smog_index(text),
            "gunning_fog": textstat.gunning_fog(text),
            "linsear_write_formula": textstat.linsear_write_formula(text),
        }

    # ── Structural / surface features ─────────────────────────────────────────

    def _structural_features(
        self, text: str, tokens: list[str], sentences: list[str]
    ) -> dict[str, float]:
        words = [t for t in tokens if t not in string.punctuation]
        word_lengths = [len(w) for w in words] if words else [0]
        char_count = len(text.replace(" ", ""))

        return {
            "num_words": len(words),
            "num_sentences": len(sentences),
            "num_chars": char_count,
            "avg_word_length": float(np.mean(word_lengths)),
            "max_word_length": float(np.max(word_lengths)),
            "avg_sentence_length": len(words) / max(len(sentences), 1),
            "num_question_marks": text.count("?"),
            "num_code_backticks": text.count("`"),
            "has_numeric": int(bool(re.search(r"\d", text))),
            "num_uppercase_words": sum(1 for w in words if w.isupper() and len(w) > 1),
        }

    # ── POS-tag features ───────────────────────────────────────────────────────

    def _pos_features(self, tokens: list[str]) -> dict[str, float]:
        tagged = self._pos_tag(tokens)
        total = len(tagged) or 1
        counts: dict[str, int] = {}

        for _, tag in tagged:
            for prefix in self.pos_tags_of_interest:
                if tag.startswith(prefix):
                    counts[prefix] = counts.get(prefix, 0) + 1

        return {
            f"pos_frac_{tag}": counts.get(tag, 0) / total
            for tag in self.pos_tags_of_interest
        }

    # ── Bloom's verb indicators ────────────────────────────────────────────────

    def _blooms_features(self, lower_text: str) -> dict[str, float]:
        features: dict[str, float] = {}
        for level, verbs in self.blooms_verbs.items():
            hit_count = sum(
                1 for v in verbs if re.search(rf"\b{re.escape(v)}\b", lower_text)
            )
            features[f"blooms_{level}_hits"] = float(hit_count)
            features[f"blooms_{level}_any"] = float(hit_count > 0)
        return features

    # ── CT keyword indicators ──────────────────────────────────────────────────

    def _ct_features(self, lower_text: str) -> dict[str, float]:
        features: dict[str, float] = {}
        for skill, kws in self.ct_keywords.items():
            slug = skill.lower().replace(" ", "_").replace("&", "and")
            hit_count = sum(
                1 for kw in kws if re.search(rf"\b{re.escape(kw)}\b", lower_text)
            )
            features[f"ct_{slug}_hits"] = float(hit_count)
            features[f"ct_{slug}_any"] = float(hit_count > 0)
        return features

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(self, text: str) -> dict[str, Any]:
        """
        Extract all features for a single question string.

        Returns
        -------
        dict[str, Any]
            Flat dict of feature_name → float value.
        """
        text = self._clean(text)
        lower_text = text.lower()
        tokens = self._tokenize_words(text)
        sentences = self._tokenize_sentences(text)

        features: dict[str, Any] = {}
        features.update(self._readability_features(text))
        features.update(self._structural_features(text, tokens, sentences))
        features.update(self._pos_features(tokens))
        features.update(self._blooms_features(lower_text))
        features.update(self._ct_features(lower_text))

        return features

    def extract_batch(self, texts: list[str], n_jobs: int = -1, batch_size: int = 250) -> pd.DataFrame:
        """
        Extract features for a list of question strings using multi-core parallelization.

        Returns
        -------
        pd.DataFrame
            One row per question, columns = feature names.
        """
        if not texts:
            return pd.DataFrame()
        try:
            from joblib import Parallel, delayed
            rows = Parallel(n_jobs=n_jobs, batch_size=batch_size)(
                delayed(self.extract)(t) for t in texts
            )
        except Exception as exc:
            logger.warning("Parallel extraction failed (%s); falling back to sequential.", exc)
            rows = [self.extract(t) for t in texts]
        return pd.DataFrame(rows)

    @property
    def feature_names(self) -> list[str]:
        """Return feature names by extracting a dummy question."""
        dummy = "Define the concept of recursion and list its base cases."
        return list(self.extract(dummy).keys())


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO)
    fe = FeatureExtractor()

    # Demo: run on the RVCE question bank
    bank_path = Path("data/raw/rvce/question_bank.json")
    if bank_path.exists():
        with open(bank_path) as f:
            bank = json.load(f)
        questions = [q["question"] for q in bank["questions"]]
        df = fe.extract_batch(questions)
        print(df.to_string())
        df.to_parquet("data/processed/rvce_features.parquet", index=False)
        print(f"\nSaved features → data/processed/rvce_features.parquet")
    else:
        sample = "Design an algorithm to sort a list of integers using merge sort."
        print(json.dumps(fe.extract(sample), indent=2))
