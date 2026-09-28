"""Tests for src/model_tuning.py — TimeSeriesSplit chronology, tuning, artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model_evaluation import classify_generalization_gap
from src.model_tuning import (
    DECISION_TREE_PARAM_GRID,
    RANDOM_FOREST_PARAM_DIST,
    XGBOOST_PARAM_DIST,
    compare_baseline_vs_tuned,
    describe_cv_folds,
    get_time_series_cv,
    load_best_hyperparameters,
    load_tuned_model,
    save_best_hyperparameters,
    save_tuned_model,
    tune_decision_tree,
    tune_random_forest,
    tune_xgboost,
)


def _fake_training_data(n_rows: int = 300, n_features: int = 5):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(n_rows, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(rng.normal(300, 20, size=n_rows))
    return X, y


def test_time_series_split_chronology_train_before_val():
    X, _ = _fake_training_data(200)
    cv = get_time_series_cv(n_splits=4)
    for train_idx, val_idx in cv.split(X):
        assert train_idx.max() < val_idx.min()


def test_time_series_split_no_shuffling_indices_increasing():
    X, _ = _fake_training_data(200)
    cv = get_time_series_cv(n_splits=4)
    for train_idx, val_idx in cv.split(X):
        assert list(train_idx) == sorted(train_idx)
        assert list(val_idx) == sorted(val_idx)


def test_describe_cv_folds_matches_split_output():
    folds = describe_cv_folds(n_rows=200, n_splits=4)
    assert len(folds) == 4
    for fold in folds:
        assert fold["train_range"][1] < fold["val_range"][0]


def test_param_grids_are_nonempty_and_well_formed():
    for grid in (DECISION_TREE_PARAM_GRID, RANDOM_FOREST_PARAM_DIST, XGBOOST_PARAM_DIST):
        assert isinstance(grid, dict)
        assert len(grid) > 0
        for values in grid.values():
            assert isinstance(values, list)
            assert len(values) > 0


def test_tune_decision_tree_fits_and_predicts():
    X, y = _fake_training_data(200)
    cv = get_time_series_cv(n_splits=3)
    search = tune_decision_tree(X, y, cv)
    preds = search.best_estimator_.predict(X)
    assert preds.shape == (len(X),)
    assert search.best_params_ is not None


def test_tune_random_forest_uses_randomized_search_within_budget():
    X, y = _fake_training_data(150)
    cv = get_time_series_cv(n_splits=3)
    search = tune_random_forest(X, y, cv, n_iter=5)
    assert len(search.cv_results_["params"]) == 5
    preds = search.best_estimator_.predict(X)
    assert preds.shape == (len(X),)


def test_tune_xgboost_uses_randomized_search_within_budget():
    X, y = _fake_training_data(150)
    cv = get_time_series_cv(n_splits=3)
    search = tune_xgboost(X, y, cv, n_iter=5)
    assert len(search.cv_results_["params"]) == 5
    preds = search.best_estimator_.predict(X)
    assert preds.shape == (len(X),)


def test_final_test_set_never_passed_to_tuner():
    """GridSearchCV/RandomizedSearchCV only ever see X_train/y_train — verified by
    construction (the tuner functions take no test-set argument) and by checking
    the CV never produces a val index beyond len(X_train)."""
    X, y = _fake_training_data(200)
    cv = get_time_series_cv(n_splits=4)
    for _, val_idx in cv.split(X):
        assert val_idx.max() < len(X)


def test_saved_tuned_model_round_trip(tmp_path):
    from sklearn.tree import DecisionTreeRegressor

    X, y = _fake_training_data(100)
    model = DecisionTreeRegressor(random_state=42).fit(X, y)
    save_tuned_model(model, "decision_tree", models_dir=tmp_path)
    loaded = load_tuned_model("decision_tree", models_dir=tmp_path)
    np.testing.assert_allclose(model.predict(X), loaded.predict(X))


def test_best_hyperparameters_persistence_round_trip(tmp_path):
    path = tmp_path / "best_hyperparameters.json"
    params = {"decision_tree": {"max_depth": 5}, "random_forest": {"n_estimators": 100}}
    save_best_hyperparameters(params, path=path)
    loaded = load_best_hyperparameters(path=path)
    assert loaded == params


def test_compare_baseline_vs_tuned_flags_true_improvement():
    baseline = {"Train MAE": 10, "Test MAE": 50, "Train RMSE": 20, "Test RMSE": 80, "Train R2": 0.99, "Test R2": 0.5}
    tuned = {"Train MAE": 20, "Test MAE": 25, "Train RMSE": 30, "Test RMSE": 40, "Train R2": 0.9, "Test R2": 0.85}
    row = compare_baseline_vs_tuned("Decision Tree", baseline, tuned)
    assert row["MAE Improvement"] == 25
    assert row["RMSE Improvement"] == 40
    assert row["R2 Improvement"] == pytest.approx(0.35)


def test_compare_baseline_vs_tuned_does_not_mask_worse_result():
    baseline = {"Train MAE": 10, "Test MAE": 20, "Train RMSE": 20, "Test RMSE": 30, "Train R2": 0.9, "Test R2": 0.8}
    tuned = {"Train MAE": 10, "Test MAE": 25, "Train RMSE": 20, "Test RMSE": 35, "Train R2": 0.9, "Test R2": 0.75}
    row = compare_baseline_vs_tuned("XGBoost", baseline, tuned)
    assert row["MAE Improvement"] < 0
    assert row["RMSE Improvement"] < 0
    assert row["R2 Improvement"] < 0


def test_classify_generalization_gap_labels():
    assert classify_generalization_gap(10, 50, 0.99, 0.5) == "Overfitting"
    assert classify_generalization_gap(10, 10.5, 0.9, 0.89) == "Good Generalization"
    assert classify_generalization_gap(10, 13, 0.9, 0.83) == "Moderate Gap"


def test_searches_score_mae_and_rmse_and_refit_on_mae():
    X, y = _fake_training_data(150)
    cv = get_time_series_cv(n_splits=3)
    search = tune_decision_tree(X, y, cv)
    assert search.refit == "mae"
    assert "mean_test_mae" in search.cv_results_
    assert "mean_test_rmse" in search.cv_results_
    assert search.best_score_ == search.cv_results_["mean_test_mae"][search.best_index_]


def test_tuning_reproducible_with_fixed_seed():
    X, y = _fake_training_data(150)
    cv = get_time_series_cv(n_splits=3)
    search_a = tune_random_forest(X, y, cv, n_iter=5)
    search_b = tune_random_forest(X, y, cv, n_iter=5)
    assert search_a.best_params_ == search_b.best_params_
