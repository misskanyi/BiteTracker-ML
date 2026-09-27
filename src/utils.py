"""Shared utilities: logging setup and reproducibility helpers."""

# In simple terms: small helper functions (consistent logging, fixed random
# seeds) that other files reuse instead of repeating themselves.

from __future__ import annotations

import logging
import random

import numpy as np

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a consistent format.

    Configures the root handler once per process; subsequent calls just
    return a named child logger.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)


def set_global_seed(seed: int) -> None:
    """Seed all RNGs used across the project for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
