from django.urls import path
from django.views.generic import RedirectView

from web.administration import views

app_name = "administration"

urlpatterns = [
    path("", RedirectView.as_view(pattern_name="administration:users"), name="index"),
    path("access-requests/", views.access_requests, name="access_requests"),
    path("access-requests/<int:pk>/", views.access_request_detail, name="access_request_detail"),
    path("users/", views.users, name="users"),
    path("users/new/", views.user_create, name="user_create"),
    path("users/<int:pk>/", views.user_detail, name="user_detail"),
    path("users/<int:pk>/edit/", views.user_edit, name="user_edit"),
    path("users/<int:pk>/toggle-active/", views.user_toggle_active, name="user_toggle_active"),
    path("users/<int:pk>/password/", views.user_password, name="user_password"),
    path("schools/", views.schools, name="schools"),
    path("schools/<int:pk>/", views.school_detail, name="school_detail"),
    path("schools/<int:pk>/edit/", views.school_edit, name="school_edit"),
    path("uploads/", views.uploads, name="uploads"),
    path("uploads/<int:pk>/", views.upload_review, name="upload_review"),
    path("logs/", views.logs, name="logs"),
    path("issues/", views.issues, name="issues"),
    path("issues/<int:pk>/", views.issue_detail, name="issue_detail"),
    path("announcements/", views.announce, name="announce"),
]
