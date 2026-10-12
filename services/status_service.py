"""What data the active model and forecasts are built on, for labelling every page.

The label comes from the saved model metadata (`results_label`) and the
configured dataset path. Anything that looks synthetic is labelled synthetic:
if either source says synthetic, the result is synthetic, so a real-looking
label can never hide synthetic results. If the metadata is unavailable the
status is "unknown" rather than assumed real.
"""

from __future__ import annotations

from config import settings
from services import ServiceError
from services.model_service import load_model_metadata

SYNTHETIC_LABEL = "SYNTHETIC DEVELOPMENT DATA"
UNKNOWN_LABEL = "DATA STATUS UNKNOWN"


def get_data_status() -> dict:
    """{kind, label, detail, results_label, dataset_name, model_name, model_trained_at}.

    `kind` is "synthetic", "real" or "unknown".
    """
    if settings.USE_GOOGLE_SHEET:
        dataset_name = settings.GOOGLE_SHEET_DATASET_NAME
        dataset_is_synthetic = False
    else:
        active = settings.ACTIVE_DATA_PATH
        dataset_name = active.name
        dataset_is_synthetic = active.resolve() == settings.SYNTHETIC_DATA_PATH.resolve()
    status = {
        "dataset_name": dataset_name,
        "results_label": None,
        "model_name": None,
        "model_trained_at": None,
    }
    try:
        metadata = load_model_metadata()
    except ServiceError:
        metadata = None

    if metadata is not None:
        status.update(
            results_label=metadata["results_label"],
            model_name=metadata["selected_model"],
            model_trained_at=metadata.get("timestamp"),
        )

    if dataset_is_synthetic or (metadata is not None and "SYNTHETIC" in metadata["results_label"].upper()):
        status.update(
            kind="synthetic",
            label=SYNTHETIC_LABEL,
            detail="Not Food4Education results. Real school data has not yet been integrated.",
        )
    elif metadata is None:
        status.update(
            kind="unknown",
            label=UNKNOWN_LABEL,
            detail="Model metadata is unavailable, so the data behind results cannot be confirmed. Treat results as unverified.",
        )
    else:
        status.update(kind="real", label=metadata["results_label"], detail=f"Model trained on {dataset_name}.")
    return status
