"""Tests for src/model_selection.py and the selection wiring in run_tuning.py.

Final model selection must depend only on TimeSeriesSplit validation scores
from the training period — never on test-set metrics.

What this file checks:
- The model with the lowest validation MAE wins; RMSE breaks ties.
- Test metrics can't change the choice: rows carrying test metrics are rejected, and the
  selection functions don't take test data.
- Empty, NaN or infinite scores are rejected.
- Linear regression is scored with the same folds and scorer as the tuned models, so
  scores are comparable.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeRegressor

import run_tuning
from src.model_selection import (
    CV_SCORING,
    SELECTION_CRITERION,
    ValidationScore,
    cross_validate_linear_regression,
    select_model_by_validation,
    validation_score_from_search,
)
from src.model_tuning import get_time_series_cv


def _score(model, cv_mae, cv_rmse=None):
    return ValidationScore(model=model, cv_mae=cv_mae, cv_rmse=cv_rmse, cv_mae_std=0.0, n_folds=5)


def _fake_training_data(n_rows=240, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n_rows, 4)), columns=list("abcd"))
    y = pd.Series(300 + 10 * X["a"] - 5 * X["b"] + rng.normal(0, 3, n_rows))
    return X, y


def test_selects_lowest_validation_mae():
    selected, ranked = select_model_by_validation(
        [_score("Random Forest", 12.0), _score("Linear Regression", 9.5), _score("XGBoost", 10.1)]
    )
    assert selected.model == "Linear Regression"
    assert [s.model for s in ranked] == ["Linear Regression", "XGBoost", "Random Forest"]


def test_validation_rmse_breaks_mae_ties():
    selected, _ = select_model_by_validation([_score("A", 10.0, cv_rmse=15.0), _score("B", 10.0, cv_rmse=13.0)])
    assert selected.model == "B"


def test_test_metrics_cannot_change_the_selected_model():
    """A has the better validation MAE but a much worse test MAE: A must still win."""
    candidates = {
        "Model A": {"validation_mae": 20.0, "test_mae": 50.0},
        "Model B": {"validation_mae": 30.0, "test_mae": 10.0},
    }
    scores = [_score(name, c["validation_mae"]) for name, c in candidates.items()]
    selected, _ = select_model_by_validation(scores)
    assert selected.model == "Model A"

    # Changing the test numbers in any way leaves the selection untouched.
    candidates["Model A"]["test_mae"] = 1e9
    candidates["Model B"]["test_mae"] = 0.0
    selected_again, _ = select_model_by_validation([_score(n, c["validation_mae"]) for n, c in candidates.items()])
    assert selected_again.model == "Model A"


def test_selector_rejects_rows_carrying_test_metrics():
    rows = [
        {"Model": "Model A", "Validation MAE": 20.0, "Test MAE": 50.0},
        {"Model": "Model B", "Validation MAE": 30.0, "Test MAE": 10.0},
    ]
    with pytest.raises(TypeError, match="ValidationScore"):
        select_model_by_validation(rows)
    with pytest.raises(TypeError):
        run_tuning.select_final_model(rows)


def test_validation_score_has_no_test_fields():
    fields = set(ValidationScore.__dataclass_fields__)
    assert not any("test" in f.lower() for f in fields)


def test_selection_function_signatures_take_no_test_data():
    for fn in (run_tuning.select_final_model, select_model_by_validation):
        params = list(inspect.signature(fn).parameters)
        assert params == [params[0]] and "test" not in params[0].lower()
    for fn in (run_tuning.tune_models, run_tuning.collect_validation_scores):
        params = inspect.signature(fn).parameters
        assert not any("test" in p.lower() for p in params)


def test_empty_or_nonfinite_scores_rejected():
    with pytest.raises(ValueError):
        select_model_by_validation([])
    with pytest.raises(ValueError):
        select_model_by_validation([_score("A", float("nan"))])


def test_linear_regression_cv_uses_same_folds_and_scorer():
    X, y = _fake_training_data()
    cv = get_time_series_cv(n_splits=4)
    score = cross_validate_linear_regression(X, y, cv)
    expected = -cross_val_score(LinearRegression(), X, y, cv=cv, scoring="neg_mean_absolute_error")
    assert score.model == "Linear Regression"
    assert score.n_folds == 4
    assert score.cv_mae == pytest.approx(expected.mean())
    assert score.cv_rmse is not None and score.cv_rmse >= score.cv_mae


def test_linear_regression_and_search_scores_are_comparable():
    """A one-point 'grid' for a model must give the same CV MAE as plain cross-validation,
    so tuned and untuned candidates are scored identically."""
    X, y = _fake_training_data()
    cv = get_time_series_cv(n_splits=4)
    search = GridSearchCV(LinearRegression(), {"fit_intercept": [True]}, scoring=CV_SCORING, refit="mae", cv=cv)
    search.fit(X, y)
    from_search = validation_score_from_search("Linear Regression", search)
    from_cv = cross_validate_linear_regression(X, y, cv)
    assert from_search.cv_mae == pytest.approx(from_cv.cv_mae)
    assert from_search.cv_rmse == pytest.approx(from_cv.cv_rmse)
    assert from_search.n_folds == from_cv.n_folds


def test_validation_score_from_single_metric_search():
    X, y = _fake_training_data()
    cv = get_time_series_cv(n_splits=3)
    search = GridSearchCV(
        DecisionTreeRegressor(random_state=0), {"max_depth": [2, 3]}, scoring="neg_mean_absolute_error", cv=cv
    ).fit(X, y)
    score = validation_score_from_search("Decision Tree", search)
    assert score.cv_mae == pytest.approx(-search.best_score_)
    assert score.cv_rmse is None
    assert score.n_folds == 3


def test_selection_criterion_wording():
    assert SELECTION_CRITERION == "lowest TimeSeriesSplit validation MAE"
