"""Regression tests for the decision-support model label.

`build_decision_support_result` used to default to "Tuned XGBoost" while the
selected model was Random Forest. The label must now come from the saved
model metadata, never from a hardcoded name.
"""

from __future__ import annotations

import json

import pytest

from config import settings
from src.decision_support import build_decision_support_result, selected_model_label


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "MODELS_DIR", tmp_path)
    return tmp_path


def _write_metadata(models_dir, **fields):
    (models_dir / "final_best_model_metadata.json").write_text(json.dumps(fields))


def _result(**kwargs):
    return build_decision_support_result(predicted_meals=400, recent_std=30, horizon_days=1, horizon_label="Next school day", **kwargs)


def test_default_label_is_selected_model_from_metadata(models_dir):
    _write_metadata(models_dir, selected_model="Random Forest")
    assert selected_model_label() == "Random Forest"
    assert _result().model_label == "Random Forest"


def test_default_label_never_tuned_xgboost_when_random_forest_selected(models_dir):
    _write_metadata(models_dir, selected_model="Random Forest")
    assert "XGBoost" not in _result().model_label


def test_default_label_follows_metadata_when_selection_changes(models_dir):
    _write_metadata(models_dir, selected_model="Decision Tree")
    assert _result().model_label == "Decision Tree"


@pytest.mark.parametrize("contents", [None, "{broken json", json.dumps({"other": 1}), json.dumps({"selected_model": ""}), json.dumps([1, 2])])
def test_unreadable_metadata_gives_neutral_label_not_a_model_name(models_dir, contents):
    if contents is not None:
        (models_dir / "final_best_model_metadata.json").write_text(contents)
    label = _result().model_label
    assert label == "Selected model (metadata unavailable)"
    assert "XGBoost" not in label and "Random Forest" not in label


def test_explicit_label_still_wins(models_dir):
    _write_metadata(models_dir, selected_model="Random Forest")
    assert _result(model_label="Custom label").model_label == "Custom label"


def test_real_metadata_label_matches_saved_selection(real_artifacts):
    assert _result().model_label == real_artifacts.metadata["selected_model"]
