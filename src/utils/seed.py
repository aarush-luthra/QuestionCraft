"""
src/utils/seed.py
=================
Deterministic seed configuration across Python, NumPy, PyTorch, and OS environment.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """
    Set seeds for full reproducibility.

    Parameters
    ----------
    seed : int
        Integer seed value (default: 42).
    """
    # Import torch BEFORE seeding: its first import can consume Python/NumPy random state,
    # which would make the first draw after set_seed() differ from later calls.
    try:
        import torch
    except ImportError:
        torch = None

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
