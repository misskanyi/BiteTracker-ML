"""Framework-independent service layer between the web UI and the ML code in src/.

Nothing in this package imports Django, so the same services
can back Django views, a REST/FastAPI endpoint or another frontend. Services
orchestrate the existing `src/` modules; they do not reimplement forecasting,
feature engineering or decision-support logic, and they never train models.

Every service error derives from `ServiceError`. Errors about a request
(invalid input, unknown school) have user-facing messages; errors about
missing or mismatched artifacts may include file paths for logs, so a UI
should replace them (see web/core/errors.py).
"""


class ServiceError(Exception):
    """Base class for errors raised by the service layer."""
