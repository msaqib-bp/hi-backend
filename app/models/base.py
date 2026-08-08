"""Declarative base and shared column types."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

#: JSON on SQLite, JSONB on PostgreSQL — same Python interface, indexable in production.
JSONColumn = JSON().with_variant(JSONB(), "postgresql")


class UTCDateTime(TypeDecorator):
    """A timestamp that is always timezone-aware UTC in Python, on every backend.

    PostgreSQL round-trips ``timestamptz`` faithfully; **SQLite has no timezone type**
    and silently returns naive datetimes. Mixing the two blows up on the first
    subtraction — which is exactly what resolution-time statistics and the overdue check
    do on every request, so the bug surfaces as a 500 rather than as a wrong number.

    Normalising in the column type means the rest of the codebase can subtract two
    timestamps without ever asking which database it is talking to.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect):  # noqa: ANN001
        if value is None:
            return None
        # A naive value here is a programming slip; assume UTC rather than guessing
        # the server's local zone, which would vary between dev and Render.
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect):  # noqa: ANN001
        if value is None:
            return None
        if value.tzinfo is None:  # SQLite
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


#: Timezone-aware timestamps everywhere. Naive datetimes make resolution-time
#: statistics quietly wrong once the server and database disagree on timezone.
TimestampColumn = UTCDateTime()


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class UUIDPrimaryKeyMixin:
    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        TimestampColumn, default=utcnow, nullable=False, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        TimestampColumn, default=utcnow, onupdate=utcnow, nullable=False
    )
