from django.apps import AppConfig


class SchoolsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "web.schools"
    label = "schools"
    verbose_name = "Schools, datasets and forecasts"
