from django.urls import path

from web.schoolcalendar import views

app_name = "schoolcalendar"

urlpatterns = [
    path("", views.calendar_view, name="calendar"),
    path("events/new/", views.event_create, name="event_create"),
    path("events/<int:pk>/", views.event_detail, name="event_detail"),
    path("events/<int:pk>/edit/", views.event_update, name="event_update"),
    path("events/<int:pk>/delete/", views.event_delete, name="event_delete"),
]
