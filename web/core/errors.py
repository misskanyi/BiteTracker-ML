"""Turn service errors into messages that are safe to show users.

Service error messages can contain file-system paths (useful in logs, not in
a browser), so configuration/artifact errors are replaced with fixed
messages here. Errors about the user's own request are shown as-is.
"""

from __future__ import annotations

import logging

from services import ServiceError
from services.data_service import DataLoadError, DataNotFoundError, UnknownSchoolError
from services.forecast_service import InsufficientHistoryError, InvalidForecastRequestError
from services.model_service import FeatureAlignmentError, ModelLoadError, ModelMetadataError, ModelNotFoundError
from services.performance_service import PerformanceDataError

logger = logging.getLogger(__name__)

_ADMIN_ACTION = "An administrator needs to run the training pipeline (scripts/train_models.py)."

_FIXED_MESSAGES = {
    ModelNotFoundError: f"The forecasting model has not been generated yet. {_ADMIN_ACTION}",
    ModelLoadError: (
        "The saved forecasting model could not be loaded. An administrator should check the installed "
        "package versions or regenerate the model."
    ),
    ModelMetadataError: (
        "The saved model's metadata is missing or does not match the model, so this feature is unavailable. "
        + _ADMIN_ACTION
    ),
    FeatureAlignmentError: (
        "The saved model does not match the features BiteTracker builds for forecasting, so no forecast was "
        "produced. " + _ADMIN_ACTION
    ),
    DataNotFoundError: f"Historical meal-demand data has not been generated yet. {_ADMIN_ACTION}",
    DataLoadError: f"Historical meal-demand data could not be read. {_ADMIN_ACTION}",
    PerformanceDataError: f"Model performance results are missing or incomplete. {_ADMIN_ACTION}",
}

# Errors describing the user's own request; their messages contain no paths.
_REQUEST_ERRORS = (InvalidForecastRequestError, UnknownSchoolError, InsufficientHistoryError)


def user_message(exc: ServiceError) -> str:
    if isinstance(exc, _REQUEST_ERRORS):
        return str(exc)
    logger.warning("Service error shown to user as a generic message: %s", exc)
    for error_type, message in _FIXED_MESSAGES.items():
        if isinstance(exc, error_type):
            return message
    return "Something went wrong while preparing this page. Please try again or contact an administrator."
