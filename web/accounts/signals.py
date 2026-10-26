"""Give every new account a profile, so role and preference lookups never miss."""

from django.conf import settings
from django.db.models.signals import post_save
from django.dispatch import receiver

from web.accounts.models import Role, UserProfile


@receiver(post_save, sender=settings.AUTH_USER_MODEL)
def create_profile(sender, instance, created, raw=False, **kwargs):
    if created and not raw:
        UserProfile.objects.get_or_create(
            user=instance, defaults={"role": Role.SYSTEM_ADMIN if instance.is_superuser else Role.SCHOOL_OPERATOR}
        )
