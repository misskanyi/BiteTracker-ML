"""What data the active model and forecasts are built on, for labelling every page.

The saved model metadata is the source of truth. Its `data_source` field
(written by `scripts/train_models.py`) records what the model was actually
trained on, so the banner is right on any machine, whether or not the Google
Sheet link is configured there (production servers normally do not have it).

Anything that looks synthetic is labelled synthetic: a synthetic data source
or a synthetic `results_label` both win, so a real-looking label can never
hide synthetic results. If the metadata is unavailable or names an unknown
source, the status is "unknown" rather than assumed real.

Older metadata written before `data_source` existed falls back to the
configured dataset path, as before.
"""

from __future__ import annotations

from config import settings
from services import ServiceError
from services.model_service import load_model_metadata
from src.feature_engineering import FOOD4EDUCATION_SOURCE, SYNTHETIC_SOURCE

SYNTHETIC_LABEL = "SYNTHETIC DEVELOPMENT DATA"
UNKNOWN_LABEL = "DATA STATUS UNKNOWN"

_SYNTHETIC_DETAIL = "Not Food4Education results. Real school data has not yet been integrated."
_DATASET_NAMES = {
    FOOD4EDUCATION_SOURCE: settings.GOOGLE_SHEET_DATASET_NAME,
    SYNTHETIC_SOURCE: settings.SYNTHETIC_DATA_PATH.name,
}


def _configured_dataset() -> tuple[str, bool]:
    """(name, is_synthetic) of the dataset configured on this machine; only used for legacy metadata."""
    if settings.USE_GOOGLE_SHEET:
        return settings.GOOGLE_SHEET_DATASET_NAME, False
    active = settings.ACTIVE_DATA_PATH
    return active.name, active.resolve() == settings.SYNTHETIC_DATA_PATH.resolve()


def get_data_status() -> dict:
    """{kind, label, detail, results_label, dataset_name, model_name, model_trained_at, data_source}.

    `kind` is "synthetic", "real" or "unknown".
    """
    try:
        metadata = load_model_metadata()
    except ServiceError:
        metadata = None

    status = {"results_label": None, "model_name": None, "model_trained_at": None, "data_source": None}
    if metadata is not None:
        status.update(
            results_label=metadata["results_label"],
            model_name=metadata["selected_model"],
            model_trained_at=metadata.get("timestamp"),
            data_source=metadata.get("data_source"),
        )

    source = status["data_source"]
    if source is not None:
        dataset_name = _DATASET_NAMES.get(source, str(source))
        dataset_is_synthetic = source == SYNTHETIC_SOURCE
        source_is_known = source in _DATASET_NAMES
    else:
        dataset_name, dataset_is_synthetic = _configured_dataset()
        source_is_known = True
    status["dataset_name"] = dataset_name

    label_is_synthetic = metadata is not None and "SYNTHETIC" in metadata["results_label"].upper()
    if dataset_is_synthetic or label_is_synthetic:
        status.update(kind="synthetic", label=SYNTHETIC_LABEL, detail=_SYNTHETIC_DETAIL)
    elif metadata is None:
        status.update(
            kind="unknown",
            label=UNKNOWN_LABEL,
            detail="Model metadata is unavailable, so the data behind results cannot be confirmed. Treat results as unverified.",
        )
    elif not source_is_known:
        status.update(
            kind="unknown",
            label=UNKNOWN_LABEL,
            detail="The model metadata names an unrecognised data source. Treat results as unverified.",
        )
    else:
        status.update(kind="real", label=metadata["results_label"], detail=f"Model trained on {dataset_name}.")
    return status
