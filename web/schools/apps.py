from django.apps import AppConfig


class SchoolsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "web.schools"
    label = "schools"
    verbose_name = "Schools, datasets and forecasts"

    def ready(self):
        # Forecasting reads approved demand history from the database (approved uploads included).
        from services import data_service
        from web.schools.demand_history import DatabaseHistoryProvider

        data_service.register_history_provider(DatabaseHistoryProvider())
