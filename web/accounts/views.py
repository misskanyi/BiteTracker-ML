"""My Profile, Settings, profile photos and the post-login landing page."""

from __future__ import annotations

from django.contrib import messages
from django.contrib.auth import update_session_auth_hash
from django.contrib.auth import views as auth_views
from django.contrib.auth.forms import PasswordChangeForm
from django.core.exceptions import PermissionDenied
from django.http import HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods, require_POST

from web.accounts import access
from web.accounts.forms import AppearanceForm, NotificationPreferencesForm, ProfileForm, ThemeForm
from web.accounts.models import UserProfile
from web.audit.models import AuditLog
from web.audit.services import log_action
from web.core.about import about_context
from web.core.uploads import image_response
from web.notifications.models import Notification
from web.notifications.services import notify

LANDING_URLS = {
    UserProfile.LandingPage.DASHBOARD: "core:dashboard",
    UserProfile.LandingPage.FORECAST: "core:forecast",
}
SETTINGS_TABS = ("appearance", "notifications", "security", "about")


def landing_url(user) -> str:
    return reverse(LANDING_URLS.get(access.get_profile(user).landing_page, "core:dashboard"))


class LoginView(auth_views.LoginView):
    """Django's login view, sending users to their chosen landing page when no ?next= is given."""

    template_name = "accounts/login.html"
    redirect_authenticated_user = True

    def get_default_redirect_url(self):
        return landing_url(self.request.user)


# --- My Profile -------------------------------------------------------------------------------


def _profile_context(request) -> dict:
    user = request.user
    return {
        "profile": access.get_profile(user),
        "role_label": access.role_label(user),
        "schools": access.accessible_schools(user) if not access.is_system_admin(user) else None,
        "recent_activity": AuditLog.objects.filter(actor=user)[:8],
    }


def profile(request):
    return render(request, "accounts/profile.html", _profile_context(request))


@require_http_methods(["GET", "POST"])
def profile_edit(request):
    user, profile_obj = request.user, access.get_profile(request.user)
    initial = {"first_name": user.first_name, "last_name": user.last_name, "phone": profile_obj.phone}
    form = ProfileForm(request.POST or None, request.FILES or None, initial=initial)
    if request.method == "POST" and form.is_valid():
        data = form.cleaned_data
        changed = [f for f in ("first_name", "last_name") if getattr(user, f) != data[f]]
        user.first_name, user.last_name = data["first_name"], data["last_name"]
        user.save(update_fields=["first_name", "last_name"])
        if profile_obj.phone != data["phone"]:
            changed.append("phone")
            profile_obj.phone = data["phone"]
        if data["avatar"]:
            if profile_obj.avatar:
                profile_obj.avatar.delete(save=False)
            profile_obj.avatar.save(data["avatar"].name, data["avatar"], save=False)
            changed.append("photo")
        elif data["remove_avatar"] and profile_obj.avatar:
            profile_obj.avatar.delete(save=False)
            changed.append("photo removed")
        profile_obj.save()
        if changed:
            log_action(user, AuditLog.Action.PROFILE_UPDATED, target=user, target_user=user, details={"fields": changed})
        messages.success(request, "Your profile has been updated.")
        return redirect("accounts:profile")
    if request.method == "POST":
        messages.error(request, "Your profile was not saved. Check the highlighted fields.")
    return render(request, "accounts/profile.html", {**_profile_context(request), "form": form, "editing": True})


def avatar(request, user_id: int):
    """Profile photos are private files: only the owner and administrators may load them."""
    target_profile = get_object_or_404(UserProfile.objects.select_related("user"), user_id=user_id)
    if request.user.pk != user_id and not (access.is_system_admin(request.user)
                                            or access.can_manage_user(request.user, target_profile.user)):
        raise PermissionDenied
    return image_response(target_profile.avatar)


# --- Settings ---------------------------------------------------------------------------------


@require_http_methods(["GET", "POST"])
def settings_view(request):
    user, profile_obj = request.user, access.get_profile(request.user)
    tab = request.GET.get("tab", "appearance")
    tab = tab if tab in SETTINGS_TABS else "appearance"
    section = request.POST.get("section") if request.method == "POST" else None

    appearance_form = AppearanceForm(request.POST if section == "appearance" else None, instance=profile_obj)
    notifications_form = NotificationPreferencesForm(request.POST if section == "notifications" else None, instance=profile_obj)
    password_form = PasswordChangeForm(user, request.POST if section == "password" else None)
    for field in password_form.fields.values():
        field.widget.attrs["class"] = "form-control"

    if section == "appearance":
        tab = "appearance"
        if appearance_form.is_valid():
            appearance_form.save()
            messages.success(request, "Appearance settings saved.")
            return redirect(f"{reverse('accounts:settings')}?tab=appearance")
    elif section == "notifications":
        tab = "notifications"
        if notifications_form.is_valid():
            notifications_form.save()
            messages.success(request, "Notification preferences saved.")
            return redirect(f"{reverse('accounts:settings')}?tab=notifications")
    elif section == "password":
        tab = "security"
        if password_form.is_valid():
            password_form.save()
            update_session_auth_hash(request, password_form.user)  # keep this session signed in
            log_action(user, AuditLog.Action.PASSWORD_CHANGED, target=user, target_user=user)
            notify([user], Notification.Type.SECURITY, "Your password was changed",
                   "Your BiteTracker password was changed. If this wasn't you, contact a System Admin immediately.",
                   url=reverse("accounts:settings") + "?tab=security", include_actor=True)
            messages.success(request, "Your password has been changed.")
            return redirect(f"{reverse('accounts:settings')}?tab=security")
    elif request.method == "POST":
        return HttpResponseBadRequest("Unknown settings section.")

    return render(request, "accounts/settings.html", {
        "tab": tab,
        "tabs": [("appearance", "Appearance", "bi-palette"), ("notifications", "Notifications", "bi-bell"),
                 ("security", "Security", "bi-shield-lock"), ("about", "About BiteTracker", "bi-info-circle")],
        "appearance_form": appearance_form,
        "notifications_form": notifications_form,
        "password_form": password_form,
        "about": about_context(),
    })


@require_POST
def theme_update(request):
    """Saves the header theme toggle so the choice follows the user to other devices and logins."""
    form = ThemeForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"ok": False, "errors": form.errors}, status=400)
    profile_obj = access.get_profile(request.user)
    profile_obj.theme = form.cleaned_data["theme"]
    profile_obj.save(update_fields=["theme", "updated_at"])
    return JsonResponse({"ok": True, "theme": profile_obj.theme})
