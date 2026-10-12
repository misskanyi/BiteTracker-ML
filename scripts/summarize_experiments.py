"""List the Colab experiment runs saved to Google Drive, from the local machine.

Reads only `metadata.json` and `metrics.json` in each run folder; nothing is
changed. The experiments folder is found in this order:

1. --experiments-dir PATH
2. the BITETRACKER_EXPERIMENTS_DIR environment variable
3. Google Drive for desktop on macOS:
   ~/Library/CloudStorage/GoogleDrive-*/My Drive/BiteTracker/experiments

Usage:
    python scripts/summarize_experiments.py
    python scripts/summarize_experiments.py --experiments-dir "/path/to/BiteTracker/experiments"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

DRIVE_SUBPATH = Path("My Drive") / "BiteTracker" / "experiments"


def find_experiments_dir(explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser()
    if os.getenv("BITETRACKER_EXPERIMENTS_DIR"):
        return Path(os.environ["BITETRACKER_EXPERIMENTS_DIR"]).expanduser()
    for drive_root in sorted((Path.home() / "Library" / "CloudStorage").glob("GoogleDrive-*")):
        candidate = drive_root / DRIVE_SUBPATH
        if candidate.is_dir():
            return candidate
    return None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def summarize_run(run_dir: Path) -> dict:
    metadata = _read_json(run_dir / "metadata.json")
    final = _read_json(run_dir / "metrics.json").get("final_model", {})
    return {
        "run_id": run_dir.name,
        "status": metadata.get("status", "no metadata.json"),
        "data_source": metadata.get("data_source", "?"),
        "commit": (metadata.get("git", {}).get("commit_sha") or "?")[:7],
        "model": final.get("selected_model", "-"),
        "validation_mae": final.get("validation_mae"),
        "test_mae": final.get("final_test_mae"),
        "missing": len(metadata.get("artifacts", {}).get("missing", [])),
    }


def _fmt(value) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "-"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List BiteTracker Colab experiment runs.")
    parser.add_argument("--experiments-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    experiments_dir = find_experiments_dir(args.experiments_dir)
    if experiments_dir is None or not experiments_dir.is_dir():
        print(
            "Experiments folder not found. Install Google Drive for desktop and sign in, or pass "
            "--experiments-dir / set BITETRACKER_EXPERIMENTS_DIR.",
            file=sys.stderr,
        )
        return 1

    runs = sorted(p for p in experiments_dir.glob("run_*") if p.is_dir())
    print(f"Experiments folder: {experiments_dir}")
    if not runs:
        print("No runs yet.")
        return 0

    header = f"{'run_id':24} {'status':10} {'data':15} {'commit':8} {'model':15} {'val MAE':>8} {'test MAE':>9} {'missing':>7}"
    print(header)
    print("-" * len(header))
    for run_dir in runs:
        r = summarize_run(run_dir)
        print(
            f"{r['run_id']:24} {r['status']:10} {r['data_source']:15} {r['commit']:8} {str(r['model']):15} "
            f"{_fmt(r['validation_mae']):>8} {_fmt(r['test_mae']):>9} {r['missing']:>7}"
        )
    print("\nFood4Education runs are PROVISIONAL (open outlier-capping and leakage issues).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
