from django.urls import path

from web.schools import views

app_name = "schools"

urlpatterns = [
    path("upload/", views.upload, name="upload"),
    path("uploads/<int:pk>/", views.upload_detail, name="upload_detail"),
]
