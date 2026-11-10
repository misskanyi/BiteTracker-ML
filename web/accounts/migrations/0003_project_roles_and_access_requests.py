"""Adopt the project-design role names and add registration access requests.

Stored role values are renamed; nobody's school assignments change:
- school_operator -> school_manager (School Manager)
- programme_admin -> planning_stakeholder (Resource Planning Stakeholder, now read-only)
- system_admin stays (System Administrator)

Every existing account stays approved: access requests are only created by Google
self-registration from now on. Reversing restores the old role values.
"""

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


RENAMES = {"school_operator": "school_manager", "programme_admin": "planning_stakeholder"}


def rename_roles(apps, schema_editor):
    UserProfile = apps.get_model("accounts", "UserProfile")
    for old, new in RENAMES.items():
        UserProfile.objects.filter(role=old).update(role=new)


def restore_roles(apps, schema_editor):
    UserProfile = apps.get_model("accounts", "UserProfile")
    for old, new in RENAMES.items():
        UserProfile.objects.filter(role=new).update(role=old)


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0002_profiles_for_existing_users'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.AlterField(
            model_name='userprofile',
            name='role',
            field=models.CharField(choices=[('school_manager', 'School Manager'), ('planning_stakeholder', 'Resource Planning Stakeholder'), ('system_admin', 'System Administrator')], default='school_manager', max_length=20),
        ),
        migrations.CreateModel(
            name='AccessRequest',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('provider', models.CharField(default='google', max_length=30)),
                ('email', models.EmailField(max_length=254)),
                ('full_name', models.CharField(blank=True, max_length=300)),
                ('requested_role', models.CharField(blank=True, choices=[('school_manager', 'School Manager'), ('planning_stakeholder', 'Resource Planning Stakeholder'), ('system_admin', 'System Administrator')], max_length=30)),
                ('applicant_note', models.CharField(blank=True, help_text='Optional: the school(s) or organisation the applicant works with.', max_length=500, verbose_name='note from applicant')),
                ('status', models.CharField(choices=[('pending', 'Pending approval'), ('approved', 'Approved'), ('rejected', 'Rejected')], db_index=True, default='pending', max_length=20)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('decided_at', models.DateTimeField(blank=True, null=True)),
                ('granted_role', models.CharField(blank=True, choices=[('school_manager', 'School Manager'), ('planning_stakeholder', 'Resource Planning Stakeholder'), ('system_admin', 'System Administrator')], max_length=30)),
                ('decision_note', models.CharField(blank=True, max_length=1000)),
                ('decided_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='+', to=settings.AUTH_USER_MODEL)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='access_request', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-created_at', '-pk'],
            },
        ),
        migrations.RunPython(rename_roles, restore_roles),
    ]
