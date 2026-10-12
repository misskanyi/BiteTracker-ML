"""Load the saved, already-trained model and its metadata; never trains anything.

The model is deserialised on first use and cached for the life of the
process, keyed on the artifact files' modification times, so repeated
forecasts reuse one in-memory model and a re-run of the training pipeline is
picked up without restarting the server.

Before a loaded model is handed out, `verify_model` checks that it matches
its metadata and that it expects exactly the features inference builds
(`src.feature_engineering.FEATURE_COLUMNS`, the same list
`src.forecasting._build_feature_row` emits). A mismatch raises instead of
producing a prediction from misaligned inputs.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import settings
from services import ServiceError
from src.feature_engineering import FEATURE_COLUMNS
from src.forecasting import load_forecasting_model

MODEL_FILENAME = "final_best_model.joblib"
METADATA_FILENAME = "final_best_model_metadata.json"
FEATURE_COLUMNS_FILENAME = "feature_columns.json"

REQUIRED_METADATA_KEYS = (
    "selected_model",
    "results_label",
    "features_used",
    "target_variable",
    "validation_mae",
    "final_test_mae",
    "final_test_rmse",
    "final_test_r2",
)

# Estimator class saved for each candidate name used by model selection.
_EXPECTED_ESTIMATOR_CLASS = {
    "Linear Regression": "LinearRegression",
    "Decision Tree": "DecisionTreeRegressor",
    "Random Forest": "RandomForestRegressor",
    "XGBoost": "XGBRegressor",
}

_RETRAIN_HINT = "Run `python scripts/train_models.py` to generate the model artifacts."


class ModelNotFoundError(ServiceError):
    """The saved model file does not exist."""


class ModelLoadError(ServiceError):
    """The saved model file exists but could not be deserialised."""


class ModelMetadataError(ServiceError):
    """The metadata is missing, unreadable, incomplete, or does not describe the loaded model."""


class FeatureAlignmentError(ServiceError):
    """The saved model does not expect the features inference builds."""


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    metadata: dict
    model_path: Path

    @property
    def selected_model_name(self) -> str:
        return self.metadata["selected_model"]

    @property
    def results_label(self) -> str:
        return self.metadata["results_label"]

    @property
    def feature_columns(self) -> list[str]:
        return list(self.metadata["features_used"])


_cache: dict[Path, tuple[tuple[int, int], LoadedModel]] = {}
_cache_lock = threading.Lock()


def model_path() -> Path:
    return settings.MODELS_DIR / MODEL_FILENAME


def metadata_path() -> Path:
    return settings.MODELS_DIR / METADATA_FILENAME


def load_model_metadata() -> dict:
    """Read and minimally validate the selected model's metadata (no model loading)."""
    path = metadata_path()
    if not path.exists():
        raise ModelMetadataError(f"Model metadata not found at {path}. {_RETRAIN_HINT}")
    try:
        with open(path) as f:
            metadata = json.load(f)
    except (OSError, ValueError) as exc:
        raise ModelMetadataError(f"Model metadata at {path} could not be read: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ModelMetadataError(f"Model metadata at {path} is not a JSON object.")
    missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing:
        raise ModelMetadataError(f"Model metadata at {path} is missing required fields: {', '.join(missing)}.")
    return metadata


def verify_model(model: Any, metadata: dict) -> None:
    """Fail clearly if the model, its metadata and the inference feature list disagree."""
    expected = list(FEATURE_COLUMNS)

    # Features: metadata, saved feature list and the model itself must all
    # equal the inference feature list, in the same order.
    if list(metadata["features_used"]) != expected:
        raise FeatureAlignmentError(
            "Model metadata lists different features from the ones inference builds "
            f"(metadata: {len(metadata['features_used'])}, inference: {len(expected)}). {_RETRAIN_HINT}"
        )
    feature_file = settings.MODELS_DIR / FEATURE_COLUMNS_FILENAME
    if feature_file.exists():
        try:
            with open(feature_file) as f:
                saved_features = json.load(f)
        except (OSError, ValueError) as exc:
            raise FeatureAlignmentError(f"{feature_file} could not be read: {exc}") from exc
        if list(saved_features) != expected:
            raise FeatureAlignmentError(f"{feature_file} does not match the inference feature list. {_RETRAIN_HINT}")
    n_features = getattr(model, "n_features_in_", None)
    if n_features is not None and n_features != len(expected):
        raise FeatureAlignmentError(
            f"Saved model expects {n_features} features but inference builds {len(expected)}. {_RETRAIN_HINT}"
        )
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None and list(feature_names) != expected:
        raise FeatureAlignmentError(
            f"Saved model was trained on different feature names/order than inference builds. {_RETRAIN_HINT}"
        )

    if metadata["target_variable"] != settings.TARGET_COLUMN:
        raise ModelMetadataError(
            f"Model metadata target '{metadata['target_variable']}' does not match '{settings.TARGET_COLUMN}'."
        )

    # Model identity: the estimator class and its hyperparameters must match
    # what the metadata says was selected.
    expected_class = _EXPECTED_ESTIMATOR_CLASS.get(metadata["selected_model"])
    if expected_class is not None and type(model).__name__ != expected_class:
        raise ModelMetadataError(
            f"Metadata says the selected model is {metadata['selected_model']}, "
            f"but the saved model is a {type(model).__name__}. {_RETRAIN_HINT}"
        )
    best_parameters = metadata.get("best_parameters") or {}
    if best_parameters and hasattr(model, "get_params"):
        params = model.get_params()
        mismatched = [
            name
            for name, value in best_parameters.items()
            if isinstance(value, (str, int, float, bool, type(None)))
            and value == value  # skip NaN, which never equals itself
            and name in params
            and params[name] != value
        ]
        if mismatched:
            raise ModelMetadataError(
                f"Saved model hyperparameters differ from its metadata ({', '.join(sorted(mismatched))}). {_RETRAIN_HINT}"
            )


def get_model() -> LoadedModel:
    """Return the verified saved model, deserialising it only on first use (or after it changes on disk)."""
    m_path, meta_path = model_path(), metadata_path()
    if not m_path.exists():
        raise ModelNotFoundError(f"No trained model found at {m_path}. {_RETRAIN_HINT}")
    if not meta_path.exists():
        raise ModelMetadataError(f"Model metadata not found at {meta_path}. {_RETRAIN_HINT}")
    key = (m_path.stat().st_mtime_ns, meta_path.stat().st_mtime_ns)

    with _cache_lock:
        cached = _cache.get(m_path)
        if cached is not None and cached[0] == key:
            return cached[1]

        metadata = load_model_metadata()
        try:
            # The existing loader in src.forecasting.
            model = load_forecasting_model()
        except FileNotFoundError as exc:
            raise ModelNotFoundError(f"No trained model found at {m_path}. {_RETRAIN_HINT}") from exc
        except Exception as exc:  # unpickling errors vary by library/version
            raise ModelLoadError(
                f"The saved model at {m_path} could not be loaded ({type(exc).__name__}: {exc}). "
                "Check that the installed package versions match requirements.txt, or regenerate it. "
                + _RETRAIN_HINT
            ) from exc
        if not hasattr(model, "predict"):
            raise ModelLoadError(f"The file at {m_path} does not contain a prediction model.")

        verify_model(model, metadata)
        loaded = LoadedModel(model=model, metadata=metadata, model_path=m_path)
        _cache[m_path] = (key, loaded)
        return loaded


def clear_model_cache() -> None:
    with _cache_lock:
        _cache.clear()
