"""Google sign-in routes, mounted at /auth/.

Only what Google sign-in needs is exposed. allauth's own login, signup, password-reset
and email pages are NOT mounted: the names allauth reverses internally point back to
BiteTracker's pages instead.

Authorised redirect URI to register in Google Cloud Console:
    https://<your domain>/auth/google/login/callback/
"""

from __future__ import annotations

from functools import wraps

from allauth.socialaccount.providers.google import views as google_views
from django.conf import settings
from django.contrib.auth.decorators import login_not_required
from django.http import Http404
from django.urls import path, reverse_lazy
from django.views.generic import RedirectView

from web.accounts import views


def _google_enabled(view):
    """404 unless Google credentials are configured, so a half-configured server never starts a flow."""

    @login_not_required
    @wraps(view)
    def wrapper(request, *args, **kwargs):
        if not settings.GOOGLE_OAUTH_ENABLED:
            raise Http404
        return view(request, *args, **kwargs)

    return wrapper


def _redirect(target, *, public=True):
    view = RedirectView.as_view(url=target)
    return login_not_required(view) if public else view


urlpatterns = [
    path("google/login/", _google_enabled(google_views.oauth2_login), name="google_login"),
    path("google/login/callback/", _google_enabled(google_views.oauth2_callback), name="google_callback"),
    path("3rdparty/login/cancelled/", views.oauth_cancelled, name="socialaccount_login_cancelled"),
    path("3rdparty/login/error/", views.oauth_error, name="socialaccount_login_error"),
    path("3rdparty/signup/", views.oauth_signup_unavailable, name="socialaccount_signup"),
    # Names allauth reverses internally, pointed at BiteTracker's own pages.
    path("3rdparty/", _redirect(reverse_lazy("accounts:settings"), public=False), name="socialaccount_connections"),
    path("login/", _redirect(reverse_lazy("accounts:login")), name="account_login"),
    path("signup/", _redirect(reverse_lazy("accounts:login")), name="account_signup"),
    path("inactive/", _redirect(reverse_lazy("accounts:access_status")), name="account_inactive"),
    path("verification-sent/", _redirect(reverse_lazy("accounts:access_status")),
         name="account_email_verification_sent"),
    path("reauthenticate/", _redirect(reverse_lazy("accounts:login")), name="account_reauthenticate"),
]
