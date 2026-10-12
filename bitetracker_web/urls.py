"""Top-level URL configuration for the BiteTracker web application."""

from django.contrib import admin
from django.urls import include, path

admin.site.site_header = "BiteTracker administration"
admin.site.site_title = "BiteTracker admin"

urlpatterns = [
    path("admin/", admin.site.urls),
    path("accounts/", include("web.accounts.urls")),
    path("data/", include("web.schools.urls")),
    path("", include("web.core.urls")),
]
