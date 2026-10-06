"""WSGI entry point for the BiteTracker web application."""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "bitetracker_web.settings")

application = get_wsgi_application()
