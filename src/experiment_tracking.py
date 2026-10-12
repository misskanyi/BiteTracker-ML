"""Save one training run's outputs into its own experiment folder.

Used by `notebooks/bitetracker_training.ipynb` (Google Colab) to copy what
`scripts/train_models.py` wrote inside a fresh clone of the repository into
`MyDrive/BiteTracker/experiments/run_YYYYMMDD_HHMMSS/`. It never trains,
changes or re-evaluates anything: it only copies files that exist and records
which expected files were missing, so a run folder never contains invented
results.

Row-level outputs (per school-day test predictions, and the actual-vs-predicted
and residual charts built from them) repeat the dataset's daily target values.
They are copied for the synthetic dataset, but left out by default for
Food4Education data so raw records are not saved to Google Drive.
"""

# In simple terms: after a training run, makes a new dated folder and copies
# the model, metrics, charts and run details into it, without ever
# overwriting an earlier run.

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

PROVISIONAL_LABEL = (
    "PROVISIONAL: produced with the current real-data methodology, which still has open issues "
    "(target outlier capping and validation leakage). Not a final validated research result."
)
SYNTHETIC_LABEL = "SYNTHETIC: development data only; says nothing about real-world accuracy."

# Packages whose versions decide whether a saved model can be reloaded and reproduced.
TRACKED_PACKAGES = [
    "numpy", "pandas", "scikit-learn", "xgboost", "statsmodels",
    "joblib", "plotly", "openpyxl", "python-dotenv", "tabulate",
]

# Files `scripts/train_models.py` is expected to write, relative to the repository root.
EXPECTED_MODEL_FILES = [
    "models/final_best_model.joblib",
    "models/final_best_model_metadata.json",
    "models/tuning_metadata.json",
    "models/feature_columns.json",
    "models/train_test_split_metadata.json",
]
EXPECTED_METRIC_FILES = [
    "reports/metrics/model_comparison.csv",
    "reports/metrics/validation_model_selection.csv",
    "reports/metrics/naive_baseline_metrics.csv",
    "reports/metrics/selected_model_vs_naive_baselines.csv",
    "reports/metrics/baseline_model_metrics.csv",
    "reports/metrics/tuned_model_metrics.csv",
    "reports/metrics/baseline_vs_tuned_comparison.csv",
]
EXPECTED_REPORT_FILES = [
    "reports/Baseline_Model_Training_Report.md",
    "reports/Hyperparameter_Tuning_Report.md",
]

_ROW_LEVEL_METRIC_MARKER = "predictions"
_ROW_LEVEL_FIGURE_PREFIXES = (
    "actual_vs_predicted",
    "residuals",
    "final_best_model_actual_vs_predicted",
    "final_best_model_residuals",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def create_run_dir(experiments_root: Path, started_at: datetime | None = None) -> Path:
    """Create a new `run_YYYYMMDD_HHMMSS` folder; never reuse an existing one."""
    started_at = started_at or utc_now()
    experiments_root.mkdir(parents=True, exist_ok=True)
    base = f"run_{started_at.strftime('%Y%m%d_%H%M%S')}"
    for suffix in [""] + [f"_{n}" for n in range(2, 100)]:
        run_dir = experiments_root / f"{base}{suffix}"
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        (run_dir / "logs").mkdir()
        (run_dir / "plots").mkdir()
        return run_dir
    raise FileExistsError(f"Could not create a new run folder under {experiments_root}")


def git_info(repo_dir: Path) -> dict:
    """Commit SHA, branch and whether the checkout has local changes."""

    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""

    status = git("status", "--porcelain")
    return {
        "commit_sha": git("rev-parse", "HEAD") or None,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD") or None,
        "commit_subject": git("log", "-1", "--format=%s") or None,
        "commit_date": git("log", "-1", "--format=%cI") or None,
        "uncommitted_changes": bool(status),
    }


def package_versions(packages: list[str] | None = None) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in packages or TRACKED_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_info() -> dict:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": package_versions(),
    }


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def _is_row_level(path: Path) -> bool:
    if path.suffix == ".csv":
        return _ROW_LEVEL_METRIC_MARKER in path.stem
    return path.name.startswith(_ROW_LEVEL_FIGURE_PREFIXES)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _csv_records(path: Path) -> list[dict] | None:
    import pandas as pd

    try:
        return json.loads(pd.read_csv(path).to_json(orient="records"))
    except (OSError, ValueError):
        return None


def collect_artifacts(repo_dir: Path, run_dir: Path, include_row_level: bool) -> dict:
    """Copy the training outputs from `repo_dir` into `run_dir`.

    Returns a manifest listing what was copied, what was expected but missing,
    and what was deliberately skipped (row-level outputs, processed data).
    """
    copied: list[str] = []
    skipped: list[str] = []

    def copy(src: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(str(dest.relative_to(run_dir)))

    final_model = repo_dir / "models" / "final_best_model.joblib"
    if final_model.exists():
        copy(final_model, run_dir / "model.joblib")
    for src in sorted((repo_dir / "models").glob("*")):
        if src.is_file() and src.suffix in {".joblib", ".json"}:
            copy(src, run_dir / "models" / src.name)

    for src in sorted((repo_dir / "reports" / "metrics").glob("*.csv")):
        if _is_row_level(src) and not include_row_level:
            skipped.append(f"reports/metrics/{src.name} (row-level)")
            continue
        copy(src, run_dir / "metrics" / src.name)

    for src in sorted((repo_dir / "reports" / "figures").glob("*.html")):
        if _is_row_level(src) and not include_row_level:
            skipped.append(f"reports/figures/{src.name} (row-level)")
            continue
        copy(src, run_dir / "plots" / src.name)

    for rel in EXPECTED_REPORT_FILES:
        src = repo_dir / rel
        if src.exists():
            copy(src, run_dir / "reports" / src.name)

    if (repo_dir / "data" / "processed" / "school_meals_processed.csv").exists():
        skipped.append("data/processed/school_meals_processed.csv (dataset rows are never saved)")

    expected = EXPECTED_MODEL_FILES + EXPECTED_METRIC_FILES + EXPECTED_REPORT_FILES
    missing = [rel for rel in expected if not (repo_dir / rel).exists()]
    if not any((repo_dir / "reports" / "figures").glob("*.html")):
        missing.append("reports/figures/*.html")

    return {"copied": copied, "missing": missing, "skipped": skipped}


def build_metrics(repo_dir: Path) -> dict:
    """Gather the saved metrics into one JSON-friendly dict (nothing is recomputed)."""
    metadata = _read_json(repo_dir / "models" / "final_best_model_metadata.json") or {}
    tables = {
        Path(rel).stem: _csv_records(repo_dir / rel)
        for rel in EXPECTED_METRIC_FILES
        if (repo_dir / rel).exists()
    }
    return {
        "final_model": {
            key: metadata.get(key)
            for key in (
                "results_label", "selected_model", "model_kind", "selection_criterion",
                "validation_mae", "validation_mae_std", "validation_rmse",
                "final_test_mae", "final_test_rmse", "final_test_r2",
                "train_records", "test_records", "train_date_range", "test_date_range",
                "data_source", "target_source_column",
            )
            if key in metadata
        },
        "final_model_metadata": metadata or None,
        "tables": tables,
    }
