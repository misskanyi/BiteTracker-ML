"""Give the five Food4Education schools weather locations in different parts of Nairobi.

Food4Education confirmed that all five schools are in Nairobi County but not where. The dataset
has no sub-county. Each school is therefore given a different, well-known Nairobi area so that
the weather advisory can be demonstrated across the city. The areas are illustrative, NOT the
schools' real locations, and are stored as unconfirmed (`location_verified=False`), so every
weather display says "approximate area, not the school's confirmed location".

    python manage.py set_nairobi_school_areas          # set the areas
    python manage.py set_nairobi_school_areas --clear  # remove them again

Replace them with real coordinates under Administration → Schools → Edit when available.
"""

from decimal import Decimal

from django.core.management.base import BaseCommand

from web.schools.models import School

# Approximate centres of well-known Nairobi areas (decimal degrees).
AREAS = {
    "SCHOOL_1": ("Westlands, Nairobi", "-1.26760", "36.81080"),
    "SCHOOL_2": ("Kibera, Nairobi", "-1.31330", "36.78800"),
    "SCHOOL_3": ("Kasarani, Nairobi", "-1.22200", "36.89900"),
    "SCHOOL_4": ("Embakasi, Nairobi", "-1.31900", "36.89400"),
    "SCHOOL_5": ("Dagoretti, Nairobi", "-1.29000", "36.74300"),
}


class Command(BaseCommand):
    help = "Set illustrative, unconfirmed weather locations in different parts of Nairobi for SCHOOL_1–SCHOOL_5."

    def add_arguments(self, parser):
        parser.add_argument("--clear", action="store_true", help="Remove the locations this command set.")

    def handle(self, *args, clear=False, **options):
        for code, (label, lat, lon) in AREAS.items():
            school = School.objects.filter(school_id=code).first()
            if school is None:
                self.stdout.write(f"{code}: no school record, skipped")
                continue
            if clear:
                if school.location_label == label and not school.location_verified:
                    school.latitude = school.longitude = None
                    school.location_label = ""
                    school.save(update_fields=["latitude", "longitude", "location_label", "updated_at"])
                    self.stdout.write(f"{code}: location removed")
                continue
            if school.location_verified:
                self.stdout.write(f"{code}: has a confirmed location, left unchanged")
                continue
            school.latitude, school.longitude, school.location_label = Decimal(lat), Decimal(lon), label
            school.location_verified = False
            school.save(update_fields=["latitude", "longitude", "location_label", "location_verified", "updated_at"])
            self.stdout.write(f"{code}: {label} ({lat}, {lon}) — approximate area")
