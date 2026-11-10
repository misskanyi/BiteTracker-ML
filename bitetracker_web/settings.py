"""Django settings for the BiteTracker web application.

Secrets and environment-specific values come from environment variables
(optionally loaded from a local, untracked `.env` file). Nothing secret is
committed to the repository.

The ML pipeline's own configuration stays in `config/settings.py`; this file
only configures the Django web layer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv

from config import settings as ml_settings

# Local secrets in .env never leak into the test suite: tests run on built-in defaults only.
if "pytest" not in sys.modules and not os.getenv("BITETRACKER_SKIP_DOTENV"):
    load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
# Writable state directory (BITETRACKER_STATE_DIR, see config/settings.py).
# Equals BASE_DIR in local development, so nothing moves.
STATE_DIR = ml_settings.STATE_DIR


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


# --- Security ----------------------------------------------------------------
DEBUG = _env_bool("DJANGO_DEBUG", True)

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "")
if not SECRET_KEY:
    if not DEBUG:
        raise ImproperlyConfigured("DJANGO_SECRET_KEY must be set when DJANGO_DEBUG is false.")
    # Local development only: never used when DEBUG is off.
    SECRET_KEY = "django-insecure-local-development-only-key"

ALLOWED_HOSTS = _env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1" if DEBUG else "")
if not DEBUG and not ALLOWED_HOSTS:
    raise ImproperlyConfigured("DJANGO_ALLOWED_HOSTS must be set when DJANGO_DEBUG is false.")
CSRF_TRUSTED_ORIGINS = _env_list("DJANGO_CSRF_TRUSTED_ORIGINS")

if not DEBUG:
    # Secure cookies need HTTPS. Set DJANGO_SECURE_COOKIES=False only to try a
    # non-DEBUG build locally over plain HTTP.
    SESSION_COOKIE_SECURE = CSRF_COOKIE_SECURE = _env_bool("DJANGO_SECURE_COOKIES", True)
    SECURE_SSL_REDIRECT = _env_bool("DJANGO_SECURE_SSL_REDIRECT", False)
    # Local health probes (deploy script, load balancer) talk plain HTTP to Gunicorn.
    SECURE_REDIRECT_EXEMPT = [r"^health/$"]
    SECURE_HSTS_SECONDS = int(os.getenv("DJANGO_SECURE_HSTS_SECONDS", "0"))
    if _env_bool("DJANGO_BEHIND_HTTPS_PROXY", False):
        SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

SESSION_COOKIE_HTTPONLY = True
X_FRAME_OPTIONS = "DENY"

# --- Applications --------------------------------------------------------------
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.humanize",
    # Google sign-in (OAuth 2.0 / OpenID Connect). Only the Google provider routes are
    # mounted (web/accounts/oauth_urls.py); allauth's own signup and password pages are not.
    "allauth",
    "allauth.account",
    "allauth.socialaccount",
    "allauth.socialaccount.providers.google",
    "web.accounts",
    "web.core",
    "web.schools",
    "web.audit",
    "web.notifications",
    "web.schoolcalendar",
    "web.support",
    "web.administration",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    # Every view requires login unless explicitly marked @login_not_required.
    "django.contrib.auth.middleware.LoginRequiredMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "allauth.account.middleware.AccountMiddleware",
    # Signs out any account that is pending approval, rejected or suspended (defence in depth:
    # such accounts are also inactive, so Django never authenticates them).
    "web.accounts.middleware.AccountStatusMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

if not DEBUG:
    # Serves collected static files in production; runserver serves them in development.
    MIDDLEWARE.insert(1, "whitenoise.middleware.WhiteNoiseMiddleware")

ROOT_URLCONF = "bitetracker_web.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "web.core.context_processors.debug",
                "web.core.context_processors.data_provenance",
                "web.core.context_processors.navigation",
                "web.core.context_processors.account",
                "web.core.context_processors.privacy_notice",
            ],
        },
    },
]

WSGI_APPLICATION = "bitetracker_web.wsgi.application"
ASGI_APPLICATION = "bitetracker_web.asgi.application"

# --- Database --------------------------------------------------------------------
# Web-application data only (users, school metadata, upload records, forecast
# audit records). Historical meal-demand data stays in the ML pipeline's CSV.
# SQLite by default; set DJANGO_DB_ENGINE=postgresql to use the DB_* variables.
if os.getenv("DJANGO_DB_ENGINE", "sqlite").lower() == "postgresql":
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "HOST": os.getenv("DB_HOST", "localhost"),
            "PORT": os.getenv("DB_PORT", "5432"),
            "NAME": os.getenv("DB_NAME", "bitetracker"),
            "USER": os.getenv("DB_USER", "bitetracker_user"),
            "PASSWORD": os.getenv("DB_PASSWORD", ""),
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            # DJANGO_SQLITE_PATH wins; otherwise the database sits in the state directory.
            "NAME": Path(os.getenv("DJANGO_SQLITE_PATH", "").strip() or str(STATE_DIR / "db.sqlite3")),
        }
    }

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Authentication ---------------------------------------------------------------
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# Users may log in with either their username or their email address.
AUTHENTICATION_BACKENDS = ["web.accounts.backends.UsernameOrEmailBackend"]

LOGIN_URL = "accounts:login"
LOGIN_REDIRECT_URL = "core:dashboard"
LOGOUT_REDIRECT_URL = "accounts:login"

# --- Google sign-in (django-allauth) ---------------------------------------------------------
# Credentials come only from the environment. With either value blank the "Continue with
# Google" button is hidden and the Google routes return 404.
GOOGLE_OAUTH_CLIENT_ID = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "").strip()
GOOGLE_OAUTH_CLIENT_SECRET = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET", "").strip()
GOOGLE_OAUTH_ENABLED = bool(GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET)

ACCOUNT_ADAPTER = "web.accounts.adapters.AccountAdapter"
SOCIALACCOUNT_ADAPTER = "web.accounts.adapters.SocialAccountAdapter"
# Nobody signs up with a password: accounts come from administrators or Google + approval.
ACCOUNT_LOGIN_METHODS = {"username", "email"}
ACCOUNT_SIGNUP_FIELDS = ["email*", "username*"]
ACCOUNT_EMAIL_VERIFICATION = "none"  # Google's verified-email claim is required instead (adapter)
ACCOUNT_UNIQUE_EMAIL = True
ACCOUNT_PREVENT_ENUMERATION = False
SOCIALACCOUNT_AUTO_SIGNUP = True
SOCIALACCOUNT_EMAIL_REQUIRED = True
SOCIALACCOUNT_EMAIL_VERIFICATION = "none"
# Never sign in to, or attach Google to, an existing account just because the email matches.
# Existing users link Google from Settings -> Security while signed in.
SOCIALACCOUNT_EMAIL_AUTHENTICATION = False
SOCIALACCOUNT_EMAIL_AUTHENTICATION_AUTO_CONNECT = False
SOCIALACCOUNT_LOGIN_ON_GET = False  # starting a sign-in needs a CSRF-protected POST
# The Google redirect URI is built from the request's host. Production always uses https://, even
# if a proxy (Cloudflare "Flexible" SSL, plain-HTTP Nginx) forwards the request over http.
ACCOUNT_DEFAULT_HTTP_PROTOCOL = os.getenv("DJANGO_OAUTH_PROTOCOL", "http" if DEBUG else "https").strip().lower()
SOCIALACCOUNT_STORE_TOKENS = False  # only identity is needed; no Google API access is kept
SOCIALACCOUNT_ONLY = False
SOCIALACCOUNT_PROVIDERS = {
    "google": {
        "APPS": [{"client_id": GOOGLE_OAUTH_CLIENT_ID, "secret": GOOGLE_OAUTH_CLIENT_SECRET, "key": ""}]
        if GOOGLE_OAUTH_ENABLED else [],
        "SCOPE": ["openid", "profile", "email"],
        "AUTH_PARAMS": {"access_type": "online", "prompt": "select_account"},
        "OAUTH_PKCE_ENABLED": True,
        "EMAIL_AUTHENTICATION": False,
    }
}

# --- Account emails ---------------------------------------------------------------------------
# A few account events (registration received/decided, new request for administrators,
# suspension/reactivation, role or school changes) are also emailed; see
# web/notifications/emails.py. Everything else stays in-app.
#
# BITETRACKER_EMAIL_DELIVERY:
#   console  - print emails to the terminal (default when DJANGO_DEBUG=True; nothing is sent)
#   disabled - record but never send (default when DJANGO_DEBUG=False until SMTP is configured)
#   smtp     - send through EMAIL_HOST (any SMTP provider: Amazon SES, Brevo, Mailgun, Google
#              Workspace SMTP relay, ...). Recipients can use Gmail or any other provider.
_EMAIL_BACKENDS = {
    "console": "django.core.mail.backends.console.EmailBackend",
    "disabled": "django.core.mail.backends.dummy.EmailBackend",
    "smtp": "django.core.mail.backends.smtp.EmailBackend",
}
EMAIL_DELIVERY = os.getenv("BITETRACKER_EMAIL_DELIVERY", "").strip().lower() or ("console" if DEBUG else "disabled")
if EMAIL_DELIVERY not in _EMAIL_BACKENDS:
    raise ImproperlyConfigured(f"BITETRACKER_EMAIL_DELIVERY must be one of: {', '.join(_EMAIL_BACKENDS)}.")
EMAIL_BACKEND = _EMAIL_BACKENDS[EMAIL_DELIVERY]
# Controlled tests: when set (comma-separated), ONLY these addresses are emailed; every other
# recipient's email is recorded as "suppressed" and never sent, even after this is cleared.
EMAIL_ALLOWED_RECIPIENTS = [a.lower() for a in _env_list("EMAIL_ALLOWED_RECIPIENTS")]

EMAIL_HOST = os.getenv("EMAIL_HOST", "").strip()
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_HOST_USER = os.getenv("EMAIL_HOST_USER", "").strip()
EMAIL_HOST_PASSWORD = os.getenv("EMAIL_HOST_PASSWORD", "").strip()
EMAIL_USE_TLS = _env_bool("EMAIL_USE_TLS", True)
EMAIL_USE_SSL = _env_bool("EMAIL_USE_SSL", False)
# Emails are sent right after the database commit, inside the web request: keep the connect/send
# timeout short so an unreachable mail server delays a page by seconds, not minutes.
EMAIL_TIMEOUT = int(os.getenv("EMAIL_TIMEOUT", "5"))
# Inside a web request, stop starting new sends after this many seconds; the rest wait for
# `send_pending_emails`. Not a hard cap: a step already started can still take up to EMAIL_TIMEOUT
# per network operation (see deploy/EMAIL.md).
EMAIL_REQUEST_BUDGET_SECONDS = float(os.getenv("EMAIL_REQUEST_BUDGET_SECONDS", "10"))
DEFAULT_FROM_EMAIL = os.getenv("DEFAULT_FROM_EMAIL", "").strip() or "BiteTracker <no-reply@localhost>"
SERVER_EMAIL = DEFAULT_FROM_EMAIL
# Retries (python manage.py send_pending_emails): attempts per email, and how old an email may be
# and still be worth sending.
EMAIL_MAX_ATTEMPTS = int(os.getenv("EMAIL_MAX_ATTEMPTS", "5"))
EMAIL_RETRY_MAX_AGE_HOURS = int(os.getenv("EMAIL_RETRY_MAX_AGE_HOURS", "48"))

if EMAIL_DELIVERY == "smtp":
    if not EMAIL_HOST or not os.getenv("DEFAULT_FROM_EMAIL", "").strip():
        raise ImproperlyConfigured("BITETRACKER_EMAIL_DELIVERY=smtp needs EMAIL_HOST and DEFAULT_FROM_EMAIL.")
    if EMAIL_USE_TLS and EMAIL_USE_SSL:
        raise ImproperlyConfigured("Set only one of EMAIL_USE_TLS (port 587) and EMAIL_USE_SSL (port 465).")
    if EMAIL_HOST_PASSWORD and EMAIL_HOST_PASSWORD == GOOGLE_OAUTH_CLIENT_SECRET:
        raise ImproperlyConfigured("EMAIL_HOST_PASSWORD must be an SMTP credential, not the Google sign-in client secret.")
    if EMAIL_HOST_USER and not EMAIL_HOST_PASSWORD:
        raise ImproperlyConfigured("EMAIL_HOST_USER is set but EMAIL_HOST_PASSWORD is empty.")
    if DEBUG and not EMAIL_ALLOWED_RECIPIENTS:
        # A local database can hold real people's addresses: local SMTP only ever emails a named allow-list.
        raise ImproperlyConfigured("Local SMTP delivery (DJANGO_DEBUG=True) needs EMAIL_ALLOWED_RECIPIENTS "
                                   "set to the address(es) you own.")


def _default_site_url() -> str:
    if DEBUG:
        return "http://localhost:8000"
    host = next((h for h in ALLOWED_HOSTS if not h.startswith((".", "*"))), "")
    return f"https://{host}" if host else ""


# Absolute address used for links in emails (no trailing slash), e.g. https://bitetracker.example.org
SITE_URL = (os.getenv("BITETRACKER_SITE_URL", "").strip() or _default_site_url()).rstrip("/")

# --- Internationalisation -------------------------------------------------------------
LANGUAGE_CODE = "en-gb"
TIME_ZONE = "Africa/Nairobi"
USE_I18N = True
USE_TZ = True

# --- Static files and uploads ------------------------------------------------------------
STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {
        # Hashed, compressed files served by WhiteNoise in production
        # (requires `python manage.py collectstatic`).
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"
        if DEBUG
        else "whitenoise.storage.CompressedManifestStaticFilesStorage"
    },
}

# Uploaded datasets are stored under <state dir>/data/uploads/ (gitignored). MEDIA_URL is
# deliberately not routed: uploaded files are never served over HTTP.
MEDIA_ROOT = ml_settings.UPLOADS_DIR
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
# Profile photos and support screenshots: PNG, JPEG or WebP only, checked by content, not just extension.
MAX_AVATAR_SIZE_BYTES = 2 * 1024 * 1024
MAX_SCREENSHOT_SIZE_BYTES = 5 * 1024 * 1024
MAX_LETTER_SIZE_BYTES = 5 * 1024 * 1024

# --- Logging -------------------------------------------------------------------------------
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"standard": {"format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "standard"}},
    "root": {"handlers": ["console"], "level": "INFO"},
    "loggers": {"django": {"handlers": ["console"], "level": "INFO", "propagate": False}},
}
