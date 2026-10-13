"""
tests/conftest.py
=================
Shared pytest fixtures for QuestionCraft test suite.
"""

import pytest
import pandas as pd


@pytest.fixture
def sample_questions():
    """Return a list of test questions representing diverse difficulty and taxonomy levels."""
    return [
        "What is the definition of a binary search tree?",
        "Explain how Dijkstra's algorithm computes the shortest path in a weighted graph.",
        "Design and implement a cache-aware B-Tree in C++ for large-scale key-value storage.",
        "Calculate the time complexity of merge sort on an array of length N.",
        "Analyze the recursion tree and solve the recurrence relation T(n) = 2T(n/2) + O(n).",
    ]


@pytest.fixture
def sample_dataframe(sample_questions):
    """Return a mock questions DataFrame."""
    return pd.DataFrame({
        "question_id": [f"q_{i}" for i in range(len(sample_questions))],
        "question_text": sample_questions,
        "difficulty": ["Easy", "Medium", "Hard", "Medium", "Hard"],
        "blooms_level": ["L1", "L2", "L6", "L3", "L4"],
        "ct_skills": [
            "Pattern Recognition",
            "Algorithmic Thinking",
            "Decomposition|Algorithmic Thinking|Abstraction",
            "Algorithmic Thinking",
            "Evaluation & Debugging|Abstraction",
        ],
    })
