from django.urls import path

from web.schools import views

app_name = "schools"

urlpatterns = [
    path("upload/", views.upload, name="upload"),
    path("uploads/<int:pk>/", views.upload_detail, name="upload_detail"),
    path("uploads/<int:pk>/file/", views.upload_file, name="upload_file"),
    path("uploads/<int:pk>/letter/", views.upload_letter, name="upload_letter"),
    path("templates/data.xlsx", views.data_template, name="data_template"),
    path("templates/supporting-letter.docx", views.letter_template, name="letter_template"),
]
