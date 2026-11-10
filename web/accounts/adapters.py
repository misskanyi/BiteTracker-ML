"""django-allauth adapters: Google sign-in with administrator approval.

The rules, in order:

1. Google must report the email address as verified, or sign-in stops.
2. A Google account already linked to a BiteTracker account signs in to that account.
3. A new Google identity whose email belongs to an existing account is NOT merged: the
   person is asked to sign in with their password and link Google in Settings -> Security.
4. Any other new Google identity creates an inactive account with a pending AccessRequest,
   and every System Administrator is notified (in-app and by email); the applicant is emailed
   that the request was received. No role or school is granted. Only the first sign-in creates
   the request, so repeat sign-ins never notify or email again.
5. Every sign-in (password or Google) passes `pre_login`: only approved, active accounts
   get a session. Pending, rejected and suspended people see the access-status page.

Credentials and tokens are never logged; tokens are not stored (SOCIALACCOUNT_STORE_TOKENS).
"""

from __future__ import annotations

import logging

from allauth.account.adapter import DefaultAccountAdapter
from allauth.core.exceptions import ImmediateHttpResponse
from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from allauth.socialaccount.providers.base import AuthProcess
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.db import transaction
from django.shortcuts import redirect
from django.urls import reverse

from web.accounts import access
from web.accounts.models import AccessRequest

logger = logging.getLogger(__name__)

# Session key naming the account whose status the access-status page shows.
STATUS_SESSION_KEY = "bt_access_status_user"

PENDING_MESSAGE = (
    "Your registration request has been sent to the administrator. "
    "You will be able to access BiteTracker once your account has been approved."
)


def show_access_status(request, user):
    """Remember who is asking (no login) and send them to the access-status page."""
    request.session[STATUS_SESSION_KEY] = user.pk
    return redirect("accounts:access_status")


def _back_to_login(request, message: str, level=messages.ERROR):
    messages.add_message(request, level, message)
    return ImmediateHttpResponse(redirect("accounts:login"))


class AccountAdapter(DefaultAccountAdapter):
    def is_open_for_signup(self, request) -> bool:
        # No username/password self-signup: administrators create accounts, or people
        # register with Google and wait for approval (SocialAccountAdapter).
        return False

    def pre_login(self, request, user, **kwargs):
        status = access.account_status(user)
        if status == access.APPROVED:
            return None
        if status in (access.REJECTED, access.SUSPENDED):
            from web.audit.models import AuditLog
            from web.audit.services import log_action

            log_action(None, AuditLog.Action.ACCESS_DENIED, target=user, target_user=user,
                       target_label=user.get_username(), details={"status": status, "method": "google"})
        return show_access_status(request, user)

    def respond_user_inactive(self, request, user):
        return show_access_status(request, user)

    def get_login_redirect_url(self, request) -> str:
        from web.accounts.views import landing_url

        return landing_url(request.user)

    def add_message(self, request, level, message_template=None, message_context=None, extra_tags="", message=None):
        # allauth's own "Successfully signed in" style messages are not shown; BiteTracker
        # adds its own where they help.
        return None


class SocialAccountAdapter(DefaultSocialAccountAdapter):
    def is_open_for_signup(self, request, sociallogin) -> bool:
        return settings.GOOGLE_OAUTH_ENABLED

    def pre_social_login(self, request, sociallogin) -> None:
        verified = [e.email for e in sociallogin.email_addresses if e.verified and e.email]
        if not verified:
            raise _back_to_login(request, "Google did not confirm that this email address is verified, "
                                          "so it can't be used to sign in to BiteTracker.")
        if sociallogin.is_existing or sociallogin.state.get("process") == AuthProcess.CONNECT:
            return  # a linked account (status checked in pre_login), or linking while signed in
        if get_user_model().objects.filter(email__iexact=verified[0]).exists():
            # Never merge by email: prove ownership of the existing account first.
            raise _back_to_login(
                request,
                "A BiteTracker account already uses this email address. Log in with your username and "
                "password, then link Google in Settings → Security to use Google next time.",
                level=messages.WARNING,
            )

    def populate_user(self, request, sociallogin, data):
        user = super().populate_user(request, sociallogin, data)
        user.is_active = False  # nothing is granted until an administrator approves
        user.is_staff = user.is_superuser = False
        return user

    @transaction.atomic
    def save_user(self, request, sociallogin, form=None):
        sociallogin.user.is_active = False
        user = super().save_user(request, sociallogin, form)
        email = next((e.email for e in sociallogin.email_addresses if e.verified), user.email)
        access_request, created = AccessRequest.objects.get_or_create(
            user=user,
            defaults={"email": email, "full_name": user.get_full_name(), "provider": sociallogin.account.provider},
        )
        if created:
            _announce_access_request(access_request)
        return user

    def get_connect_redirect_url(self, request, socialaccount) -> str:
        return reverse("accounts:settings") + "?tab=security"

    def on_authentication_error(self, request, provider, error=None, exception=None, extra_context=None):
        # Log the kind of failure only: provider responses can contain codes and tokens.
        logger.warning("Google sign-in failed: error=%s exception=%s", error,
                       type(exception).__name__ if exception else None)
        if error == "cancelled":
            raise _back_to_login(request, "Google sign-in was cancelled.", level=messages.INFO)
        raise _back_to_login(request, "Google sign-in could not be completed. Please try again.")


def _announce_access_request(access_request: AccessRequest) -> None:
    from web.audit.models import AuditLog
    from web.audit.services import log_action
    from web.notifications import emails
    from web.notifications.models import Notification
    from web.notifications.services import notify, system_admins

    user = access_request.user
    log_action(None, AuditLog.Action.ACCESS_REQUESTED, target=access_request, target_user=user,
               target_label=access_request.email, details={"provider": access_request.provider})
    who = access_request.full_name or access_request.email
    notify(system_admins(), Notification.Type.ACCESS_REQUEST, "New registration request",
           f"{who} ({access_request.email}) is waiting for account approval.",
           url=reverse("administration:access_request_detail", args=[access_request.pk]),
           event_key=f"access_request:{access_request.pk}:submitted")
    # Emails go out only after the registration is committed; one per recipient.
    emails.registration_received(access_request)
    emails.access_request_for_admins(access_request)
