"""Structured logging.

JSON in production (so Render's log viewer and any log shipper can parse it), coloured
console output in development. A request-id is bound to every log line emitted while
handling a request, which makes tracing a single complaint submission through
API -> manager -> AI -> database straightforward.
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar

import structlog

from app.core.config import settings

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def new_request_id() -> str:
    rid = uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


def current_request_id() -> str:
    return _request_id.get()


def _inject_request_id(_logger, _method, event_dict):  # noqa: ANN001
    event_dict["request_id"] = current_request_id()
    return event_dict


def configure_logging() -> None:
    """Wire structlog and the stdlib logging module together. Idempotent."""
    level = logging.DEBUG if settings.DEBUG else logging.INFO

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level, force=True)

    # Uvicorn ships its own handlers; let ours own the output instead.
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(noisy).handlers.clear()
        logging.getLogger(noisy).propagate = True

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _inject_request_id,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.is_production:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None):
    return structlog.get_logger(name)
