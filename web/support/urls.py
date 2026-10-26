from django.urls import path

from web.support import views

app_name = "support"

urlpatterns = [
    path("", views.help_page, name="help"),
    path("reports/<int:pk>/", views.issue_detail, name="issue_detail"),
    path("reports/<int:pk>/screenshot/", views.issue_screenshot, name="issue_screenshot"),
]
