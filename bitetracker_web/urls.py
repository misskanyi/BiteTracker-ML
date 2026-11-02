"""Top-level URL configuration for the BiteTracker web application."""

from django.contrib import admin
from django.urls import include, path

from web.core.health import health

admin.site.site_header = "BiteTracker administration"
admin.site.site_title = "BiteTracker admin"

urlpatterns = [
    path("health/", health, name="health"),
    path("admin/", admin.site.urls),
    path("accounts/", include("web.accounts.urls")),
    path("data/", include("web.schools.urls")),
    path("calendar/", include("web.schoolcalendar.urls")),
    path("notifications/", include("web.notifications.urls")),
    path("help/", include("web.support.urls")),
    path("administration/", include("web.administration.urls")),
    path("", include("web.core.urls")),
]
