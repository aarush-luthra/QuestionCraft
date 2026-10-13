"""
tests/test_feature_engineering.py
=================================
Unit tests for FeatureExtractor class and linguistic / semantic feature extraction.
"""

import pytest
from src.data.feature_engineering import FeatureExtractor


@pytest.fixture(scope="module")
def extractor():
    return FeatureExtractor()


def test_feature_extractor_initialization(extractor):
    """Verify that FeatureExtractor initializes with valid dictionaries."""
    assert len(extractor.blooms_verbs) > 0
    assert "L1" in extractor.blooms_verbs
    assert "L6" in extractor.blooms_verbs
    assert len(extractor.ct_keywords) > 0
    assert "Algorithmic Thinking" in extractor.ct_keywords


def test_feature_extraction_vector_length(extractor):
    """Verify that feature extraction yields a consistent non-empty dictionary."""
    text = "Explain the difference between a stack and a queue with an example."
    feats = extractor.extract(text)

    assert isinstance(feats, dict)
    assert len(feats) > 25

    # Check key feature groups
    assert "flesch_reading_ease" in feats
    assert "num_words" in feats
    assert "avg_word_length" in feats
    assert "pos_frac_NN" in feats
    assert "blooms_L2_hits" in feats
    assert "ct_algorithmic_thinking_hits" in feats


def test_blooms_verb_detection(extractor):
    """Test action verb detection for Bloom's Taxonomy."""
    # L1: define
    l1_text = "Define a linked list."
    feats_l1 = extractor.extract(l1_text)
    assert feats_l1["blooms_L1_hits"] >= 1
    assert feats_l1["blooms_L1_any"] == 1.0

    # L6: design
    l6_text = "Design a distributed consensus protocol."
    feats_l6 = extractor.extract(l6_text)
    assert feats_l6["blooms_L6_hits"] >= 1
    assert feats_l6["blooms_L6_any"] == 1.0


def test_ct_keyword_detection(extractor):
    """Test keyword detection for Computational Thinking skills."""
    text = "Write an algorithm to decompose the matrix and sort each partition."
    feats = extractor.extract(text)

    assert feats["ct_algorithmic_thinking_hits"] >= 1
    assert feats["ct_decomposition_hits"] >= 1


def test_empty_string_handling(extractor):
    """Ensure graceful extraction on empty strings."""
    feats = extractor.extract("")
    assert isinstance(feats, dict)
    assert feats["num_words"] == 0
    assert feats["num_sentences"] == 0
