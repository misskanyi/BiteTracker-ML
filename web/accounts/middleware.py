"""Signs out accounts that are not approved, whatever route they reach.

Pending, rejected and suspended accounts are inactive, so Django already refuses to
authenticate them. This middleware is the second line: if any such account ever ends
up with a session (for example it was suspended mid-session, or a status was changed
outside the application), the session is ended before any view runs.
"""

from __future__ import annotations

from django.contrib.auth import logout
from django.http import JsonResponse

from web.accounts import access
from web.accounts.adapters import show_access_status


class AccountStatusMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        user = getattr(request, "user", None)
        if user is not None and user.is_authenticated and not access.is_approved(user):
            logout(request)
            if request.headers.get("Accept", "").startswith("application/json"):
                return JsonResponse({"error": "account_not_active"}, status=401)
            return show_access_status(request, user)
        return self.get_response(request)
