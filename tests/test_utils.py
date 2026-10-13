"""
tests/test_utils.py
===================
Unit tests for configuration loading, seed setting, and dataset hashing.
"""

from pathlib import Path
import pandas as pd
from src.utils.config import load_config
from src.utils.hashing import compute_dataframe_md5
from src.utils.seed import set_seed


def test_load_config():
    """Verify that model_config.yaml loads with expected root keys."""
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "global" in cfg
    assert "tfidf" in cfg
    assert "difficulty_classifier" in cfg
    assert "blooms_classifier" in cfg
    assert "ct_classifier" in cfg
    assert "duplicate_detector" in cfg


def test_seed_setting():
    """Verify that set_seed executes without errors."""
    set_seed(42)
    import random
    import numpy as np
    val1 = random.random()
    val2 = np.random.rand()
    set_seed(42)
    assert random.random() == val1
    assert np.random.rand() == val2


def test_dataframe_md5_determinism(sample_dataframe):
    """Verify that dataframe hashing is deterministic and handles list/dict columns."""
    h1 = compute_dataframe_md5(sample_dataframe)
    h2 = compute_dataframe_md5(sample_dataframe.copy())
    assert h1 == h2
    assert len(h1) == 32  # standard 128-bit MD5 hex string
