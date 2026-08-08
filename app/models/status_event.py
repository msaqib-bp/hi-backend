"""Audit trail of complaint lifecycle changes.

Two things depend on this table:

1. **Resolution-time statistics** (a Batch 4 requirement) need timestamped transitions,
   not just a single ``resolved_at`` column — this is what lets us measure how long a
   complaint sat unassigned versus how long the actual repair took.
2. **The citizen tracking timeline** — "Received → Assigned to Water Supply → In
   Progress → Resolved", each with a date, which is the whole point of giving people a
   reference code.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Enum as SAEnum
from sqlalchemy import ForeignKey, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.models.enums import ComplaintStatus

if TYPE_CHECKING:
    from app.models.complaint import Complaint


class StatusEvent(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "status_events"

    complaint_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("complaints.id", ondelete="CASCADE"), index=True
    )
    complaint: Mapped[Complaint] = relationship(back_populates="events", lazy="noload")

    from_status: Mapped[ComplaintStatus | None] = mapped_column(
        SAEnum(
            ComplaintStatus,
            native_enum=False,
            length=24,
            values_callable=lambda enum: [m.value for m in enum],
        )
    )
    to_status: Mapped[ComplaintStatus] = mapped_column(
        SAEnum(
            ComplaintStatus,
            native_enum=False,
            length=24,
            values_callable=lambda enum: [m.value for m in enum],
        ),
        nullable=False,
    )

    note: Mapped[str | None] = mapped_column(Text)
    #: Free-text actor label ("system", "Municipal Administrator") so the timeline reads
    #: naturally without forcing a join on every render.
    actor: Mapped[str] = mapped_column(String(160), default="system", nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<StatusEvent {self.from_status} -> {self.to_status}>"
