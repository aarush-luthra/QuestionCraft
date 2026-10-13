"""
src/utils/config.py
===================
YAML configuration loader and path resolver.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("questioncraft.utils.config")

DEFAULT_CONFIG_PATH = Path("configs/model_config.yaml")


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """
    Load YAML configuration safely.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logger.debug("Loaded configuration from %s", path)
    return cfg or {}
