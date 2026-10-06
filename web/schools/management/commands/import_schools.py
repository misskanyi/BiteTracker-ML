"""Create or update School records from the active development dataset.

Idempotent: schools are matched on their stable `school_id`, so running the
command again updates changed fields and never creates duplicates. Only
fields present in the dataset are written; `verified_enrolment` is never set
here because dataset enrolment figures are not verified enrolment.
"""

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from services import ServiceError, data_service, status_service
from web.schools.models import School


class Command(BaseCommand):
    help = "Create or update School records from the active dataset (safe to run repeatedly)."

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Report what would change without saving.")

    def handle(self, *args, dry_run=False, **options):
        try:
            schools = data_service.list_schools()
        except ServiceError as exc:
            raise CommandError(str(exc)) from exc
        data_label = status_service.get_data_status()["label"]

        created = updated = unchanged = 0
        dataset_ids = set()
        with transaction.atomic():
            for record in schools:
                dataset_ids.add(record["school_id"])
                values = {
                    "name": record.get("school_name") or "",
                    "county": record.get("county") or "",
                    "sub_county": record.get("sub_county") or "",
                    "data_source": data_label,
                }
                school = School.objects.filter(school_id=record["school_id"]).first()
                if school is None:
                    created += 1
                    if not dry_run:
                        School.objects.create(school_id=record["school_id"], **values)
                    continue
                changed = {k: v for k, v in values.items() if getattr(school, k) != v}
                if not changed:
                    unchanged += 1
                    continue
                updated += 1
                if not dry_run:
                    for field, value in changed.items():
                        setattr(school, field, value)
                    school.save(update_fields=[*changed, "updated_at"])

        not_in_dataset = School.objects.exclude(school_id__in=dataset_ids).count()
        prefix = "[dry run] " if dry_run else ""
        self.stdout.write(
            self.style.SUCCESS(
                f"{prefix}Schools in dataset ({data_label}): {len(schools)} — "
                f"created: {created}, updated: {updated}, unchanged (skipped): {unchanged}."
            )
        )
        if not_in_dataset:
            self.stdout.write(
                f"{not_in_dataset} existing school record(s) are not in the active dataset; they were left unchanged."
            )
