"""Give every new account a profile, and audit Google account links."""

from allauth.socialaccount.signals import social_account_added
from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver

from web.accounts.models import Role, UserProfile


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_profile(sender, instance, created, raw=False, **kwargs):
    if created and not raw:
        UserProfile.objects.get_or_create(
            user=instance, defaults={"role": Role.SYSTEM_ADMIN if instance.is_superuser else Role.SCHOOL_MANAGER}
        )


@receiver(social_account_added)
def record_google_link(sender, request, sociallogin, **kwargs):
    """An existing, signed-in user linked their Google account (Settings -> Security)."""
    from web.audit.models import AuditLog
    from web.audit.services import log_action

    user = sociallogin.user
    log_action(user, AuditLog.Action.GOOGLE_LINKED, target=user, target_user=user,
               details={"provider": sociallogin.account.provider})
