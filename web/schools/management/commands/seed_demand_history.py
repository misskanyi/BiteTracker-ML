"""Copy the processed dataset into the demand-history table as the approved baseline.

Usage:
    python manage.py seed_demand_history            # only if no baseline exists yet
    python manage.py seed_demand_history --replace  # after a model release ships a new processed CSV

Until a baseline is seeded, forecasting reads the processed CSV directly (plus approved uploads).
"""

from django.core.management.base import BaseCommand, CommandError

from web.schools.demand_history import seed_baseline


class Command(BaseCommand):
    help = "Seed the approved demand-history baseline from the processed dataset."

    def add_arguments(self, parser):
        parser.add_argument("--replace", action="store_true", help="Replace an existing baseline.")

    def handle(self, *args, replace: bool = False, **options):
        try:
            written = seed_baseline(replace=replace)
        except FileNotFoundError as exc:
            raise CommandError(str(exc)) from exc
        if written:
            self.stdout.write(self.style.SUCCESS(f"Seeded {written} approved baseline rows."))
        else:
            self.stdout.write("A baseline already exists; nothing written (use --replace to swap it).")
