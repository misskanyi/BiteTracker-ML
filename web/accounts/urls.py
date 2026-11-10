"""Login/logout, My Profile and Settings.

Login uses Django's built-in view (exempt from LoginRequiredMiddleware) with a
landing-page preference. LogoutView only accepts POST, so logging out always
goes through a CSRF-protected form.
"""

from django.contrib.auth import views as auth_views
from django.urls import path

from web.accounts import views

app_name = "accounts"

urlpatterns = [
    path("login/", views.LoginView.as_view(), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("access/", views.access_status, name="access_status"),
    path("access/leave/", views.access_status_leave, name="access_status_leave"),
    path("profile/", views.profile, name="profile"),
    path("profile/edit/", views.profile_edit, name="profile_edit"),
    path("avatar/<int:user_id>/", views.avatar, name="avatar"),
    path("settings/", views.settings_view, name="settings"),
    path("settings/theme/", views.theme_update, name="theme"),
]
