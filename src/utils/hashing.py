"""
src/utils/hashing.py
====================
Cryptographic dataset hashing utilities for MLflow lineage and DVC verification.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import pandas as pd


def compute_dataframe_md5(df: pd.DataFrame) -> str:
    """
    Compute a deterministic MD5 hex-digest of a pandas DataFrame.
    Safely converts list/dict columns to strings before hashing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    str
        MD5 hexadecimal checksum.
    """
    hashable_df = df.copy()
    for col in hashable_df.columns:
        if hashable_df[col].apply(lambda x: isinstance(x, (list, set, dict))).any():
            hashable_df[col] = hashable_df[col].astype(str)
    return hashlib.md5(
        pd.util.hash_pandas_object(hashable_df, index=True).values.tobytes()
    ).hexdigest()


def compute_file_md5(path: str | Path) -> str:
    """
    Compute MD5 hex-digest of a file chunk by chunk.

    Parameters
    ----------
    path : str | Path
        Path to the target file.

    Returns
    -------
    str
        MD5 hexadecimal checksum.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
