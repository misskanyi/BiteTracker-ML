"""Create (or reset) one demo account per role, with the same credentials everywhere.

Idempotent: accounts are matched on username. Running the command again resets
each account's password, role, active flag and school assignments to the values
below, so local and deployed environments end up with identical logins.

    python manage.py seed_demo_accounts                    # default password
    python manage.py seed_demo_accounts --password '...'   # or DEMO_ACCOUNT_PASSWORD
    python manage.py seed_demo_accounts --schools KE001 KE002
    python manage.py seed_demo_accounts --remove           # delete the demo accounts

Accounts created here have no AccessRequest, so they count as administrator-created
and approved. Run `import_schools` first so there are schools to assign.
"""

from __future__ import annotations

import os

from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from web.accounts import access
from web.accounts.models import Role
from web.schools.models import School

DEFAULT_PASSWORD = "BiteTracker@2026"
DEFAULT_SCHOOL_COUNT = 3

DEMO_ACCOUNTS = [
    {"username": "demo_admin", "first_name": "Demo", "last_name": "Administrator",
     "email": "demo.admin@example.com", "role": Role.SYSTEM_ADMIN},
    {"username": "demo_manager", "first_name": "Demo", "last_name": "Manager",
     "email": "demo.manager@example.com", "role": Role.SCHOOL_MANAGER},
    {"username": "demo_stakeholder", "first_name": "Demo", "last_name": "Stakeholder",
     "email": "demo.stakeholder@example.com", "role": Role.PLANNING_STAKEHOLDER},
]


class Command(BaseCommand):
    help = "Create or reset one demo account per role (System Administrator, School Manager, Stakeholder)."

    def add_arguments(self, parser):
        parser.add_argument("--password", help="Password for all demo accounts "
                            "(default: $DEMO_ACCOUNT_PASSWORD, else the built-in demo password).")
        parser.add_argument("--schools", nargs="+", metavar="SCHOOL_ID",
                            help=f"School IDs to assign to the manager and stakeholder "
                                 f"(default: the first {DEFAULT_SCHOOL_COUNT} schools).")
        parser.add_argument("--remove", action="store_true", help="Delete the demo accounts instead.")

    def handle(self, *args, password=None, schools=None, remove=False, **options):
        User = get_user_model()
        usernames = [a["username"] for a in DEMO_ACCOUNTS]

        if remove:
            deleted, _ = User.objects.filter(username__in=usernames).delete()
            self.stdout.write(self.style.SUCCESS(f"Removed demo accounts ({deleted} rows deleted)."))
            return

        password = password or os.environ.get("DEMO_ACCOUNT_PASSWORD") or DEFAULT_PASSWORD
        try:
            validate_password(password)
        except ValidationError as exc:
            raise CommandError("Password rejected: " + " ".join(exc.messages)) from exc

        school_list = self._schools(schools)

        with transaction.atomic():
            for spec in DEMO_ACCOUNTS:
                user, created = User.objects.get_or_create(username=spec["username"])
                user.first_name, user.last_name, user.email = spec["first_name"], spec["last_name"], spec["email"]
                user.is_active = True
                user.set_password(password)
                user.save()
                access.set_role(user, spec["role"])
                # Admins see every school through their role; only the other roles need assignments.
                assigned = [] if spec["role"] == Role.SYSTEM_ADMIN else school_list
                access.set_school_assignments(user, assigned, assigned_by=None)
                self.stdout.write(
                    f"{'Created' if created else 'Reset'} {spec['username']:<17} {Role(spec['role']).label}"
                    + (f" — schools: {', '.join(s.school_id for s in assigned)}" if assigned else "")
                )

        if not school_list:
            self.stdout.write(self.style.WARNING(
                "No schools assigned: run `import_schools` first, then rerun this command."
            ))
        self.stdout.write(self.style.SUCCESS("Demo accounts ready. Log in with the username or email above."))

    def _schools(self, codes: list[str] | None) -> list[School]:
        if codes:
            found = list(School.objects.filter(school_id__in=codes))
            missing = set(codes) - {s.school_id for s in found}
            if missing:
                raise CommandError(f"Unknown school IDs: {', '.join(sorted(missing))}")
            return found
        return list(School.objects.order_by("school_id")[:DEFAULT_SCHOOL_COUNT])
