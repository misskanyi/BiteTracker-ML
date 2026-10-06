"""Login/logout using Django's built-in authentication views.

LoginView is already exempt from LoginRequiredMiddleware. LogoutView only
accepts POST, so logging out always goes through a CSRF-protected form.
"""

from django.contrib.auth import views as auth_views
from django.urls import path

app_name = "accounts"

urlpatterns = [
    path(
        "login/",
        auth_views.LoginView.as_view(template_name="accounts/login.html", redirect_authenticated_user=True),
        name="login",
    ),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
]
