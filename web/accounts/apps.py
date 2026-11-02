from django.apps import AppConfig


class AccountsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "web.accounts"
    label = "accounts"
    verbose_name = "Accounts"

    def ready(self):
        from web.accounts import signals  # noqa: F401  (registers the profile signal)
