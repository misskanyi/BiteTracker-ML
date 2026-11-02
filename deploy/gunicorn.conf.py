"""Gunicorn settings for BiteTracker in production (used by deploy/systemd/bitetracker.service).

Every value can be overridden with an environment variable from
/etc/bitetracker/bitetracker.env, so the file itself never needs editing on the server.
"""

import os

# Nginx is the only public entry point; Gunicorn listens on loopback only.
bind = os.getenv("GUNICORN_BIND", "127.0.0.1:8000")

# Each worker holds its own copy of the Random Forest in memory (~tens of MB),
# so keep this modest on small instances: 2 suits a t3.small (2 GB RAM).
workers = int(os.getenv("GUNICORN_WORKERS", "2"))
threads = int(os.getenv("GUNICORN_THREADS", "2"))
worker_class = "gthread"

# Forecast and performance pages can take a few seconds on first load.
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 5

# Recycle workers now and then to keep memory use flat.
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = 100

# Trust X-Forwarded-* headers only from Nginx on the same machine.
forwarded_allow_ips = "127.0.0.1"

# Logs go to stdout/stderr, collected by journald (`journalctl -u bitetracker`).
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")

proc_name = "bitetracker"
