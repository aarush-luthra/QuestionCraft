"""
tests/test_models.py
====================
Smoke and logic unit tests for classifiers, pipelines, and duplicate detector.
"""

import pytest
from src.models.blooms_classifier import normalise_label, VALID_LABELS
from src.models.ct_classifier import parse_ct_labels, CT_LABELS


def test_blooms_label_normalization():
    """Verify that varied label string formats normalize to canonical L1-L6."""
    assert normalise_label("1") == "L1"
    assert normalise_label("remember") == "L1"
    assert normalise_label("L2") == "L2"
    assert normalise_label("understand") == "L2"
    assert normalise_label("apply") == "L3"
    assert normalise_label("analyze") == "L4"
    assert normalise_label("evaluate") == "L5"
    assert normalise_label("create") == "L6"
    assert normalise_label("unknown_garbage") is None


def test_ct_label_parsing():
    """Verify parsing of pipe-separated CT skill strings."""
    raw = "Decomposition|Algorithmic Thinking|InvalidSkill"
    parsed = parse_ct_labels(raw)
    assert "Decomposition" in parsed
    assert "Algorithmic Thinking" in parsed
    assert "InvalidSkill" not in parsed
    assert len(parsed) == 2


def test_ct_label_parsing_empty():
    """Verify handling of empty or NA CT labels."""
    assert parse_ct_labels("") == []
    assert parse_ct_labels(None) == []


def test_model_config_hyperparameters():
    """Verify that model_config.yaml specifies required model hyperparameters."""
    from src.utils.config import load_config
    cfg = load_config()

    diff_lr = cfg["difficulty_classifier"]["logistic_regression"]
    assert "C" in diff_lr
    assert "max_iter" in diff_lr

    blooms_lr = cfg["blooms_classifier"]["logistic_regression"]
    assert "C" in blooms_lr
    assert blooms_lr["multi_class"] == "multinomial"

    ct_cfg = cfg["ct_classifier"]
    assert len(ct_cfg["labels"]) == 5
