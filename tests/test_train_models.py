"""Tests for the one-command pipeline (scripts/train_models.py).

The end-to-end test runs every stage on a small synthetic dataset with cheap
stand-in tuners and writes all artifacts to a temporary directory, so it
never touches the real models/ or reports/ folders and avoids the full
hyperparameter search.

What this file checks:
- Argument parsing works and the data path resolves (configured default, or a CLI path
  made absolute).
- The full pipeline runs every stage and saves the artifacts.
- Model selection happens before test evaluation and never sees test rows.
- When a stage fails, the error names the stage; the CLI exits with an error if the
  dataset is missing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

import run_tuning
from config import settings
from scripts import train_models
from src import pipeline
from src.model_tuning import tune_random_forest, tune_xgboost
from src.synthetic_data import generate_synthetic_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHEAP_TUNERS = {
    "decision_tree": lambda X, y, cv: GridSearchCV(
        DecisionTreeRegressor(random_state=settings.RANDOM_SEED),
        {"max_depth": [3, 5]},
        scoring="neg_mean_absolute_error",
        cv=cv,
    ).fit(X, y),
    "random_forest": lambda X, y, cv: tune_random_forest(X, y, cv, n_iter=1),
    "xgboost": lambda X, y, cv: tune_xgboost(X, y, cv, n_iter=1),
}


@pytest.fixture
def isolated_outputs(tmp_path, monkeypatch):
    """Point every output directory at tmp_path and swap in cheap tuners."""
    dirs = {
        "PROCESSED_DATA_DIR": tmp_path / "processed",
        "MODELS_DIR": tmp_path / "models",
        "REPORTS_DIR": tmp_path / "reports",
        "METRICS_DIR": tmp_path / "reports" / "metrics",
        "FIGURES_DIR": tmp_path / "reports" / "figures",
    }
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(settings, name, path)
    monkeypatch.setattr(run_tuning, "TUNERS", CHEAP_TUNERS)
    return dirs


@pytest.fixture
def small_dataset(tmp_path):
    df = generate_synthetic_dataset(start_date="2023-01-01", end_date="2023-06-30")
    path = tmp_path / "small_synthetic.csv"
    df.to_csv(path, index=False)
    return path


def test_parse_args_defaults_to_no_data_path():
    assert train_models.parse_args([]).data_path is None


def test_parse_args_accepts_data_path():
    assert train_models.parse_args(["--data-path", "some/file.csv"]).data_path == Path("some/file.csv")


def test_resolve_data_path_uses_configured_active_path_by_default():
    assert train_models.resolve_data_path(None) == settings.ACTIVE_DATA_PATH


def test_no_real_dataset_configured_stops_instead_of_using_synthetic(monkeypatch, capsys):
    monkeypatch.setattr(settings, "USE_GOOGLE_SHEET", False)
    monkeypatch.delenv("BITETRACKER_DATA_PATH", raising=False)
    assert train_models.main([]) == 2
    assert "BITETRACKER_GOOGLE_SHEET_URL" in capsys.readouterr().err


def test_resolve_data_path_makes_cli_path_absolute():
    resolved = train_models.resolve_data_path(Path("some/file.csv"))
    assert resolved.is_absolute()
    assert resolved.name == "file.csv"


def test_full_pipeline_runs_all_stages_and_saves_artifacts(isolated_outputs, small_dataset, capsys):
    exit_code = train_models.main(["--data-path", str(small_dataset)])
    assert exit_code == 0

    out = capsys.readouterr().out
    for n in range(1, train_models.TOTAL_STAGES + 1):
        assert f"[{n}/{train_models.TOTAL_STAGES}]" in out
    # Stages must print in order.
    positions = [out.index(f"[{n}/{train_models.TOTAL_STAGES}]") for n in range(1, train_models.TOTAL_STAGES + 1)]
    assert positions == sorted(positions)

    models_dir = isolated_outputs["MODELS_DIR"]
    metrics_dir = isolated_outputs["METRICS_DIR"]
    assert (isolated_outputs["PROCESSED_DATA_DIR"] / "school_meals_processed.csv").exists()
    for name in ("linear_regression", "decision_tree", "random_forest", "xgboost"):
        assert (models_dir / f"{name}.joblib").exists()
    for name in ("decision_tree", "random_forest", "xgboost"):
        assert (models_dir / f"{name}_tuned.joblib").exists()
    assert (models_dir / "final_best_model.joblib").exists()
    assert (metrics_dir / "baseline_model_metrics.csv").exists()
    assert (metrics_dir / "tuned_model_metrics.csv").exists()
    for name in (
        "naive_baseline_metrics.csv",
        "naive_baseline_test_predictions.csv",
        "validation_model_selection.csv",
        "model_comparison.csv",
        "selected_model_vs_naive_baselines.csv",
    ):
        assert (metrics_dir / name).exists(), name

    metadata = json.loads((models_dir / "final_best_model_metadata.json").read_text())
    assert metadata["selected_model"] in {"Linear Regression", "Decision Tree", "Random Forest", "XGBoost"}
    assert metadata["target_variable"] == settings.TARGET_COLUMN
    assert metadata["selection_criterion"] == "lowest TimeSeriesSplit validation MAE"
    assert metadata["cv_n_folds"] == settings.CV_N_SPLITS
    assert "TimeSeriesSplit" in metadata["cv_strategy"]
    for key in ("validation_mae", "final_test_mae", "final_test_rmse", "final_test_r2", "features_used"):
        assert metadata[key] is not None, key
    assert "SYNTHETIC" in metadata["results_label"]
    # Chronological split: training must end before testing starts.
    assert metadata["train_date_range"]["end"] < metadata["test_date_range"]["start"]

    # The selected model is the validation winner, and all four candidates were validated.
    validation = pd.read_csv(metrics_dir / "validation_model_selection.csv")
    assert set(validation["Model"]) == {"Linear Regression", "Decision Tree", "Random Forest", "XGBoost"}
    assert validation.loc[validation["Validation MAE"].idxmin(), "Model"] == metadata["selected_model"]
    assert metadata["validation_mae"] == pytest.approx(validation["Validation MAE"].min())

    comparison = pd.read_csv(metrics_dir / "model_comparison.csv")
    naive_rows = comparison[comparison["Type"] == "Naive baseline"]
    assert set(naive_rows["Method"]) == {"Persistence / Last Observation", "7-Day Historical Mean"}
    assert naive_rows["Validation MAE"].isna().all()  # not fabricated
    assert "projected_students" not in " ".join(comparison["Method"])


def test_selection_happens_before_test_evaluation_and_never_sees_test_rows(
    isolated_outputs, small_dataset, monkeypatch
):
    calls = []
    real_tune, real_collect = run_tuning.tune_models, run_tuning.collect_validation_scores
    real_select, real_eval = run_tuning.select_final_model, run_tuning.evaluate_selected_model
    real_eval_all = run_tuning.evaluate_tuned_models

    def spy_tune(X_train, y_train, n_splits=5):
        calls.append(("tune", X_train.index))
        return real_tune(X_train, y_train, n_splits)

    def spy_collect(searches, X_train, y_train, n_splits=5):
        calls.append(("collect", X_train.index))
        return real_collect(searches, X_train, y_train, n_splits)

    def spy_select(validation_scores):
        calls.append(("select", None))
        return real_select(validation_scores)

    def spy_eval(selected, model, kind, X_train, y_train, X_test, y_test):
        calls.append(("evaluate_selected", X_test.index))
        return real_eval(selected, model, kind, X_train, y_train, X_test, y_test)

    def spy_eval_all(searches, X_train, y_train, X_test, *args):
        calls.append(("evaluate_all", X_test.index))
        return real_eval_all(searches, X_train, y_train, X_test, *args)

    monkeypatch.setattr(run_tuning, "tune_models", spy_tune)
    monkeypatch.setattr(run_tuning, "collect_validation_scores", spy_collect)
    monkeypatch.setattr(run_tuning, "select_final_model", spy_select)
    monkeypatch.setattr(run_tuning, "evaluate_selected_model", spy_eval)
    monkeypatch.setattr(run_tuning, "evaluate_tuned_models", spy_eval_all)

    assert train_models.main(["--data-path", str(small_dataset)]) == 0

    order = [name for name, _ in calls]
    assert order == ["tune", "collect", "select", "evaluate_selected", "evaluate_all"]
    test_index = dict(calls)["evaluate_selected"]
    for stage in ("tune", "collect"):
        assert dict(calls)[stage].intersection(test_index).empty, f"{stage} saw test rows"


def test_stage_failure_propagates_and_names_the_stage(isolated_outputs, small_dataset, monkeypatch, capsys):
    def broken_engineer_features(df):
        raise RuntimeError("feature engineering exploded")

    # Stage 3 builds features through src.pipeline (fit-on-training-period preprocessing).
    monkeypatch.setattr(pipeline, "engineer_features", broken_engineer_features)
    with pytest.raises(RuntimeError, match="feature engineering exploded"):
        train_models.main(["--data-path", str(small_dataset)])
    assert "PIPELINE FAILED during [3/9]" in capsys.readouterr().err
    assert not (isolated_outputs["MODELS_DIR"] / "final_best_model.joblib").exists()


def test_cli_exits_non_zero_when_dataset_missing(tmp_path):
    """Run the real script as CI would: a missing dataset must give a non-zero exit code."""
    result = subprocess.run(
        [sys.executable, "scripts/train_models.py", "--data-path", str(tmp_path / "missing.csv")],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert "PIPELINE FAILED during [1/9]" in result.stderr
    assert "FileNotFoundError" in result.stderr
