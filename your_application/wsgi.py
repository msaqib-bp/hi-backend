"""Compatibility shim for Render's *placeholder* start command.

WHAT THIS IS
------------
When a Render service is created as a plain "Web Service" instead of from a Blueprint,
it keeps Render's default start command::

    gunicorn your_application.wsgi

``your_application`` is a placeholder Render expects you to replace. It cannot work for
any project, and the resulting ``ModuleNotFoundError: No module named 'your_application'``
gives no hint that the *start command* is the thing that needs changing.

This module makes that command resolve to the real application, so a misconfigured
service boots correctly instead of crash-looping.

THIS IS NOT A DEGRADED PATH
---------------------------
``gunicorn.conf.py`` sets ``worker_class = uvicorn.workers.UvicornWorker``, so the app is
served as ASGI with full lifespan support — schema creation, model loading and demo
seeding all run normally. This is the same server, worker and application object the
documented start command produces; only the module path differs.

THE PROPER FIX
--------------
Set the start command (Render -> Settings -> Start Command)::

    gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:$PORT

or, better, recreate the service from ``render.yaml`` as a Blueprint, which also
provisions PostgreSQL — without it the service runs on SQLite on an ephemeral disk and
loses all data on every restart and redeploy.

Once either is done this package can be deleted. It is kept because a deployment that
works is worth more than a repository that is tidy, and removing it would silently break
any service still using the placeholder command.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.main import app

log = get_logger(__name__)

log.warning(
    "started_via_compatibility_shim",
    detail=(
        "Serving through 'your_application.wsgi', Render's placeholder start command. "
        "The app is running correctly, but set the start command to "
        "'gunicorn app.main:app -k uvicorn.workers.UvicornWorker -w 1 -b 0.0.0.0:$PORT' "
        "- or redeploy from render.yaml as a Blueprint to also get PostgreSQL, since "
        "this service is otherwise using an ephemeral SQLite file."
    ),
)

#: Gunicorn looks for `application` when the app URI carries no ":name" suffix.
application = app
