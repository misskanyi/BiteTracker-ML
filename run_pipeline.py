"""CLI entry point: run the full BiteTracker data pipeline.

Usage:
    python run_pipeline.py
"""

from __future__ import annotations

from src.pipeline import run_data_pipeline
from src.utils import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    df = run_data_pipeline()
    logger.info("Pipeline finished. Final shape: %s", df.shape)
