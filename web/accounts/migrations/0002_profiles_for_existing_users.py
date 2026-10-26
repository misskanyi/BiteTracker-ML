"""Give accounts that existed before roles a profile, without reducing anyone's access.

- Superusers become System Admins (they already had full access).
- Staff accounts become Programme Admins; everyone else becomes a School Operator.
- Before this release every signed-in user could see every school, so each existing
  non-superuser is assigned all current schools. Administrators can narrow this
  afterwards in Administration → User management.

Reversing removes only the rows this migration created (profiles and assignments).
"""

from django.conf import settings
from django.db import migrations


def forwards(apps, schema_editor):
    User = apps.get_model(*settings.AUTH_USER_MODEL.split("."))
    UserProfile = apps.get_model("accounts", "UserProfile")
    SchoolAssignment = apps.get_model("accounts", "SchoolAssignment")
    School = apps.get_model("schools", "School")
    schools = list(School.objects.all())
    for user in User.objects.all():
        if UserProfile.objects.filter(user=user).exists():
            continue
        if user.is_superuser:
            role = "system_admin"
        elif user.is_staff:
            role = "programme_admin"
        else:
            role = "school_operator"
        UserProfile.objects.create(user=user, role=role)
        if not user.is_superuser:
            SchoolAssignment.objects.bulk_create(
                [SchoolAssignment(user=user, school=school) for school in schools], ignore_conflicts=True
            )


def backwards(apps, schema_editor):
    apps.get_model("accounts", "SchoolAssignment").objects.all().delete()
    apps.get_model("accounts", "UserProfile").objects.all().delete()


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0001_initial"),
        ("schools", "0001_initial"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [migrations.RunPython(forwards, backwards)]
