"""Tests for the Colab experiment-output helpers and the local run summary script.

Everything runs on a fake repository layout in a temporary directory, so the
real models/ and reports/ folders are never read or written.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import summarize_experiments
from src import experiment_tracking as et

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = PROJECT_ROOT / "notebooks" / "bitetracker_training.ipynb"


def _fake_repo(root: Path, complete: bool = True) -> Path:
    """A repo-shaped folder holding what scripts/train_models.py would write."""
    (root / "models").mkdir(parents=True)
    (root / "reports" / "metrics").mkdir(parents=True)
    (root / "reports" / "figures").mkdir(parents=True)
    (root / "data" / "processed").mkdir(parents=True)
    (root / "models" / "final_best_model.joblib").write_bytes(b"model")
    (root / "models" / "final_best_model_metadata.json").write_text(json.dumps({
        "selected_model": "Decision Tree", "validation_mae": 18.7, "final_test_mae": 11.7,
    }))
    (root / "reports" / "metrics" / "model_comparison.csv").write_text("Method,Test MAE\nDecision Tree,11.7\n")
    (root / "reports" / "metrics" / "tuned_test_predictions.csv").write_text("date,actual\n2025-01-01,10\n")
    (root / "reports" / "figures" / "cv_score_comparison.html").write_text("<html></html>")
    (root / "reports" / "figures" / "actual_vs_predicted_xgboost.html").write_text("<html></html>")
    (root / "data" / "processed" / "school_meals_processed.csv").write_text("date\n2025-01-01\n")
    if complete:
        for rel in et.EXPECTED_MODEL_FILES + et.EXPECTED_METRIC_FILES + et.EXPECTED_REPORT_FILES:
            path = root / rel
            if not path.exists():
                path.write_text("{}" if path.suffix == ".json" else "x\n1\n")
    return root


def test_create_run_dir_never_reuses_a_folder(tmp_path):
    started = datetime(2026, 10, 12, 9, 30, 5, tzinfo=timezone.utc)
    first = et.create_run_dir(tmp_path, started)
    second = et.create_run_dir(tmp_path, started)

    assert first.name == "run_20261012_093005"
    assert second.name == "run_20261012_093005_2"
    assert (first / "logs").is_dir() and (first / "plots").is_dir()


def test_collect_artifacts_skips_row_level_outputs_for_real_data(tmp_path):
    repo = _fake_repo(tmp_path / "repo")
    run_dir = et.create_run_dir(tmp_path / "experiments")

    manifest = et.collect_artifacts(repo, run_dir, include_row_level=False)

    assert (run_dir / "model.joblib").read_bytes() == b"model"
    assert (run_dir / "metrics" / "model_comparison.csv").exists()
    assert (run_dir / "plots" / "cv_score_comparison.html").exists()
    assert not (run_dir / "metrics" / "tuned_test_predictions.csv").exists()
    assert not (run_dir / "plots" / "actual_vs_predicted_xgboost.html").exists()
    assert not list(run_dir.rglob("school_meals_processed.csv"))
    assert any("tuned_test_predictions.csv" in s for s in manifest["skipped"])
    assert manifest["missing"] == []


def test_collect_artifacts_keeps_row_level_outputs_when_allowed(tmp_path):
    repo = _fake_repo(tmp_path / "repo")
    run_dir = et.create_run_dir(tmp_path / "experiments")

    et.collect_artifacts(repo, run_dir, include_row_level=True)

    assert (run_dir / "metrics" / "tuned_test_predictions.csv").exists()
    assert (run_dir / "plots" / "actual_vs_predicted_xgboost.html").exists()
    assert not list(run_dir.rglob("school_meals_processed.csv"))


def test_missing_artifacts_are_reported_not_invented(tmp_path):
    repo = _fake_repo(tmp_path / "repo", complete=False)
    run_dir = et.create_run_dir(tmp_path / "experiments")

    manifest = et.collect_artifacts(repo, run_dir, include_row_level=True)

    assert "models/tuning_metadata.json" in manifest["missing"]
    assert "reports/Hyperparameter_Tuning_Report.md" in manifest["missing"]
    assert not (run_dir / "models" / "tuning_metadata.json").exists()


def test_build_metrics_reads_saved_results_only(tmp_path):
    repo = _fake_repo(tmp_path / "repo")

    metrics = et.build_metrics(repo)

    assert metrics["final_model"] == {"selected_model": "Decision Tree", "validation_mae": 18.7, "final_test_mae": 11.7}
    assert metrics["tables"]["model_comparison"] == [{"Method": "Decision Tree", "Test MAE": 11.7}]
    assert "tuned_test_predictions" not in metrics["tables"]


def test_git_info_records_commit_sha(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "a.txt").write_text("a")
    subprocess.run(git + ["add", "a.txt"], cwd=repo, check=True)
    subprocess.run(git + ["commit", "-qm", "first"], cwd=repo, check=True)

    info = et.git_info(repo)

    assert len(info["commit_sha"]) == 40
    assert info["uncommitted_changes"] is False
    (repo / "a.txt").write_text("changed")
    assert et.git_info(repo)["uncommitted_changes"] is True


def test_summarize_experiments_lists_runs(tmp_path, capsys):
    run_dir = et.create_run_dir(tmp_path, datetime(2026, 10, 12, tzinfo=timezone.utc))
    et.write_json(run_dir / "metadata.json", {
        "status": "completed", "data_source": "synthetic", "git": {"commit_sha": "abcdef1234"},
        "artifacts": {"missing": []},
    })
    et.write_json(run_dir / "metrics.json", {
        "final_model": {"selected_model": "Decision Tree", "validation_mae": 18.7, "final_test_mae": 11.7},
    })

    assert summarize_experiments.main(["--experiments-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "run_20261012_000000" in out and "abcdef1" in out and "18.70" in out


def test_summarize_experiments_without_folder_fails_cleanly(tmp_path):
    assert summarize_experiments.main(["--experiments-dir", str(tmp_path / "missing")]) == 1


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text())


def _source(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def test_notebook_is_valid_and_has_no_secrets():
    nb = _notebook()
    text = NOTEBOOK.read_text()

    assert nb["nbformat"] == 4 and nb["metadata"]["kernelspec"]["name"] == "python3"
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells and all(not c["outputs"] for c in code_cells)
    for cell in code_cells:
        compile(_source(cell), "<notebook cell>", "exec")
    for marker in ("ghp_", "github_pat_", "/spreadsheets/d/1"):
        assert marker not in text
    assert any('DATA_SOURCE = "synthetic"' in _source(c) for c in code_cells)


@pytest.mark.parametrize("heading", [
    "1. Project setup", "2. GitHub repository access", "3. Dependency installation",
    "4. Google Drive mounting", "5. Dataset source configuration", "6. Dataset loading and validation",
    "7. Training pipeline execution", "8. Model evaluation", "9. Saving experiment outputs",
    "10. Experiment summary",
])
def test_notebook_has_required_sections(heading):
    cells = _notebook()["cells"]
    assert any(f"## {heading}" in _source(c) for c in cells if c["cell_type"] == "markdown")
