"""Gunicorn configuration.

Gunicorn auto-loads this file from the working directory, so these settings apply on
**every** deployment — including one whose start command was never configured.

That matters on Render. A service created as a plain Web Service (rather than from
`render.yaml`) keeps Render's placeholder start command, `gunicorn your_application.wsgi`,
which supplies neither a worker class nor a bind address. Without this file that means a
sync WSGI worker (wrong — the app is ASGI) bound to 127.0.0.1 (unreachable — Render needs
0.0.0.0 on $PORT). With it, the deployment is correct either way.

Explicit flags on the command line still win, so the `render.yaml` and Dockerfile start
commands — which pass `-k`, `-w` and `-b` — behave exactly as before.
"""

from __future__ import annotations

import os

# ASGI, not WSGI. FastAPI cannot be served by gunicorn's default sync worker.
worker_class = "uvicorn.workers.UvicornWorker"

# Render routes to $PORT and requires 0.0.0.0; gunicorn's default of 127.0.0.1:8000
# would start cleanly and then be reported as "no open ports detected".
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# One worker: each loads its own copy of the scikit-learn models, so memory scales with
# worker count and two would risk an OOM restart on a 512 MB free instance.
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))

# Model loading and demo seeding happen during startup; the default 30s timeout can kill
# the first boot before it finishes.
timeout = 120
graceful_timeout = 30

# Free instances sleep and wake behind a proxy; a keep-alive slightly above the typical
# idle-connection window avoids needless reconnects.
keepalive = 5

accesslog = "-"
errorlog = "-"
