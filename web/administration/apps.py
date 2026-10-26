from django.apps import AppConfig


class AdministrationConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "web.administration"
    label = "administration"
    verbose_name = "Administration"
