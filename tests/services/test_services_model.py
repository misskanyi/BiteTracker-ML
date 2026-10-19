"""Stage 2: model_service — loading, caching, metadata checks and feature alignment.

What this file checks:
- The model is loaded once and cached, and reloaded if the file on disk changes.
- Clear errors for a missing or corrupt model and for missing or bad metadata.
- Strict feature checks: missing, reordered or wrong-count features block prediction.
- Model class, hyperparameters and target must match the metadata.
- Forecasting never trains a model; the services import without Django or training code.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import pytest

import src.forecasting
from config import settings
from services import ServiceError, forecast_service, model_service
from services.model_service import (
    FeatureAlignmentError,
    ModelLoadError,
    ModelMetadataError,
    ModelNotFoundError,
)
from src.feature_engineering import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def count_model_loads(monkeypatch):
    """Count deserialisations of the model file (src.forecasting's joblib.load)."""
    calls = []
    real_load = joblib.load

    def counting_load(path, *args, **kwargs):
        calls.append(Path(path))
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(src.forecasting.joblib, "load", counting_load)
    return calls


# --- Loading and caching ---------------------------------------------------------


def test_get_model_returns_saved_model_with_metadata(stub_artifacts):
    loaded = model_service.get_model()
    assert type(loaded.model).__name__ == "LinearRegression"
    assert loaded.selected_model_name == "Linear Regression"
    assert loaded.results_label == "SYNTHETIC DEVELOPMENT RESULTS"
    assert loaded.feature_columns == FEATURE_COLUMNS
    assert loaded.model_path == stub_artifacts.model_path


def test_model_is_deserialised_once_and_cached(stub_artifacts, count_model_loads):
    first = model_service.get_model()
    second = model_service.get_model()
    third = model_service.get_model()
    assert first is second is third
    assert len(count_model_loads) == 1


def test_repeated_forecasts_reuse_cached_model(stub_artifacts, count_model_loads):
    for horizon in (1, 7, 30):
        forecast_service.generate_forecast("SCH001", horizon)
    assert len(count_model_loads) == 1


def test_model_reloaded_after_artifact_changes_on_disk(stub_artifacts, count_model_loads):
    model_service.get_model()
    stat = stub_artifacts.model_path.stat()
    os.utime(stub_artifacts.model_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    model_service.get_model()
    assert len(count_model_loads) == 2


# --- Missing / unreadable artifacts ---------------------------------------------


def test_missing_model_raises_clear_error(stub_artifacts):
    stub_artifacts.model_path.unlink()
    with pytest.raises(ModelNotFoundError, match="scripts/train_models.py"):
        model_service.get_model()


def test_missing_metadata_raises_clear_error(stub_artifacts):
    stub_artifacts.metadata_path.unlink()
    with pytest.raises(ModelMetadataError, match="metadata not found"):
        model_service.get_model()
    with pytest.raises(ModelMetadataError, match="metadata not found"):
        model_service.load_model_metadata()


def test_unreadable_metadata_json_raises_clear_error(stub_artifacts):
    stub_artifacts.metadata_path.write_text("{not valid json")
    with pytest.raises(ModelMetadataError, match="could not be read"):
        model_service.get_model()


def test_metadata_missing_required_fields_raises(stub_artifacts, make_metadata):
    metadata = make_metadata()
    del metadata["selected_model"], metadata["final_test_mae"]
    stub_artifacts.write_metadata(metadata)
    with pytest.raises(ModelMetadataError, match="final_test_mae, selected_model|selected_model.*final_test_mae"):
        model_service.get_model()


def test_corrupt_model_file_raises_model_load_error(stub_artifacts):
    stub_artifacts.model_path.write_bytes(b"this is not a joblib file")
    with pytest.raises(ModelLoadError, match="could not be loaded"):
        model_service.get_model()


def test_file_without_predict_raises_model_load_error(stub_artifacts):
    joblib.dump({"not": "a model"}, stub_artifacts.model_path)
    with pytest.raises(ModelLoadError, match="does not contain a prediction model"):
        model_service.get_model()


# --- Feature alignment ---------------------------------------------------------------


def test_metadata_with_missing_feature_raises(stub_artifacts, make_metadata):
    stub_artifacts.write_metadata(make_metadata(features_used=FEATURE_COLUMNS[:-1]))
    with pytest.raises(FeatureAlignmentError, match="metadata: 19, inference: 20"):
        model_service.get_model()


def test_metadata_with_reordered_features_raises(stub_artifacts, make_metadata):
    reordered = [FEATURE_COLUMNS[1], FEATURE_COLUMNS[0], *FEATURE_COLUMNS[2:]]
    stub_artifacts.write_metadata(make_metadata(features_used=reordered))
    with pytest.raises(FeatureAlignmentError):
        model_service.get_model()


def test_model_expecting_wrong_feature_count_raises(stub_artifacts, make_model):
    stub_artifacts.write_model(make_model(FEATURE_COLUMNS[:-1]))
    with pytest.raises(FeatureAlignmentError, match="expects 19 features but inference builds 20"):
        model_service.get_model()


def test_model_trained_on_different_feature_order_raises(stub_artifacts, make_model):
    stub_artifacts.write_model(make_model(list(reversed(FEATURE_COLUMNS))))
    with pytest.raises(FeatureAlignmentError, match="different feature names/order"):
        model_service.get_model()


def test_saved_feature_columns_file_mismatch_raises(stub_artifacts):
    stub_artifacts.feature_columns_path.write_text(json.dumps(FEATURE_COLUMNS + ["extra_feature"]))
    with pytest.raises(FeatureAlignmentError, match="feature_columns.json"):
        model_service.get_model()


def test_feature_mismatch_blocks_prediction(stub_artifacts, make_model, monkeypatch):
    """A misaligned model must fail before any prediction is made."""
    model = make_model(FEATURE_COLUMNS[:-1])
    stub_artifacts.write_model(model)
    predictions = []
    monkeypatch.setattr(type(model), "predict", lambda self, X: predictions.append(X) or [0.0])
    with pytest.raises(FeatureAlignmentError):
        forecast_service.generate_forecast("SCH001", 1)
    assert predictions == []


# --- Model / metadata consistency ------------------------------------------------------


def test_model_class_must_match_metadata(stub_artifacts, make_metadata):
    stub_artifacts.write_metadata(make_metadata(selected_model="Random Forest"))
    with pytest.raises(ModelMetadataError, match="Random Forest.*LinearRegression"):
        model_service.get_model()


def test_model_hyperparameters_must_match_metadata(stub_artifacts, make_metadata):
    stub_artifacts.write_metadata(make_metadata(best_parameters={"fit_intercept": False}))
    with pytest.raises(ModelMetadataError, match="fit_intercept"):
        model_service.get_model()


def test_target_variable_must_match(stub_artifacts, make_metadata):
    stub_artifacts.write_metadata(make_metadata(target_variable="meals_served"))
    with pytest.raises(ModelMetadataError, match="target"):
        model_service.get_model()


def test_all_model_errors_are_service_errors():
    for error in (ModelNotFoundError, ModelLoadError, ModelMetadataError, FeatureAlignmentError):
        assert issubclass(error, ServiceError)


# --- Real saved model (skips if the pipeline has not been run) ----------------------------


def test_real_saved_model_loads_and_matches_its_metadata(real_artifacts):
    loaded = model_service.get_model()
    assert loaded.selected_model_name == real_artifacts.metadata["selected_model"]
    assert loaded.results_label == "SYNTHETIC DEVELOPMENT RESULTS"
    expected_class = model_service._EXPECTED_ESTIMATOR_CLASS[real_artifacts.metadata["selected_model"]]
    assert type(loaded.model).__name__ == expected_class


def test_real_model_training_inference_feature_alignment(real_artifacts):
    """Regression guard: the saved model, its metadata, the saved feature list and the
    features inference builds must be the same 20 columns in the same order."""
    loaded = model_service.get_model()
    saved_feature_file = json.loads((settings.MODELS_DIR / "feature_columns.json").read_text())
    assert len(FEATURE_COLUMNS) == 20
    assert loaded.model.n_features_in_ == 20
    assert list(loaded.model.feature_names_in_) == FEATURE_COLUMNS
    assert real_artifacts.metadata["features_used"] == FEATURE_COLUMNS
    assert saved_feature_file == FEATURE_COLUMNS


# --- No training, no web framework ---------------------------------------------------------


def test_forecasting_through_services_never_fits_a_model(real_artifacts, monkeypatch):
    import sklearn.ensemble
    import sklearn.linear_model
    import sklearn.tree
    import xgboost

    def forbidden_fit(self, *args, **kwargs):
        raise AssertionError(f"{type(self).__name__}.fit was called during a forecast")

    for cls in (
        sklearn.ensemble.RandomForestRegressor,
        sklearn.tree.DecisionTreeRegressor,
        sklearn.linear_model.LinearRegression,
        xgboost.XGBRegressor,
    ):
        monkeypatch.setattr(cls, "fit", forbidden_fit)

    for horizon in (1, 7, 30):
        assert forecast_service.generate_forecast("SCH001", horizon).days


def test_services_import_without_django_streamlit_or_training_code():
    code = (
        "import sys\n"
        "import services, services.data_service, services.model_service, "
        "services.forecast_service, services.performance_service\n"
        "bad = [m for m in ('django', 'streamlit', 'src.model_training', 'src.model_tuning') if m in sys.modules]\n"
        "print(','.join(bad))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "DJANGO_SETTINGS_MODULE"}
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
