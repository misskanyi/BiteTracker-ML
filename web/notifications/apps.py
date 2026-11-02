from django.apps import AppConfig


class NotificationsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "web.notifications"
    label = "notifications"
    verbose_name = "Notifications"
