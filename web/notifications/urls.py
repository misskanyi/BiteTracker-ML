from django.urls import path

from web.notifications import views

app_name = "notifications"

urlpatterns = [
    path("", views.notification_list, name="list"),
    path("feed/", views.feed, name="feed"),
    path("<int:pk>/read/", views.mark_read, name="mark_read"),
    path("<int:pk>/open/", views.open_notification, name="open"),
    path("read-all/", views.mark_all_read, name="mark_all_read"),
    path("<int:pk>/delete/", views.delete_notification, name="delete"),
    path("clear/", views.clear_all, name="clear_all"),
]
