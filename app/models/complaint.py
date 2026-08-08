"""The Complaint entity — the centre of the domain model.

This class carries behaviour, not just columns: it knows how to compute its own age and
resolution time, whether it has breached its priority's service-level target, and how to
absorb a result from the AI layer. Keeping that logic on the entity means the statistics
service and the API never re-derive it inconsistently.
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Enum as SAEnum
from sqlalchemy import Float, ForeignKey, Index, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import (
    Base,
    JSONColumn,
    TimestampColumn,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
    utcnow,
)
from app.models.enums import ComplaintCategory, ComplaintPriority, ComplaintStatus

if TYPE_CHECKING:
    from app.models.department import Department
    from app.models.status_event import StatusEvent

#: Ambiguity-free alphabet — no 0/O, no 1/I/L. Reference codes get read out over the
#: phone and typed by hand into the tracking page.
_REFERENCE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"


def generate_reference_code() -> str:
    """Return a short human-friendly tracking code, e.g. ``CIV-8F3K2A``."""
    suffix = "".join(secrets.choice(_REFERENCE_ALPHABET) for _ in range(6))
    return f"CIV-{suffix}"


def _enum_column(enum_cls, length: int):
    """Portable enum column: a native ENUM on PostgreSQL is painful to migrate, so
    store the value as VARCHAR with a CHECK constraint instead."""
    return SAEnum(
        enum_cls,
        native_enum=False,
        length=length,
        values_callable=lambda enum: [member.value for member in enum],
    )


class Complaint(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "complaints"
    __table_args__ = (
        # The admin queue is almost always filtered by status and sorted by recency.
        Index("ix_complaints_status_created", "status", "created_at"),
        Index("ix_complaints_category_priority", "category", "priority"),
    )

    # ------------------------------------------------------------- identity
    reference_code: Mapped[str] = mapped_column(
        String(16), unique=True, index=True, default=generate_reference_code, nullable=False
    )

    # ------------------------------------------------ citizen-supplied input
    description: Mapped[str] = mapped_column(Text, nullable=False)
    location: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    reporter_name: Mapped[str | None] = mapped_column(String(160))
    reporter_contact: Mapped[str | None] = mapped_column(String(160))
    image_url: Mapped[str | None] = mapped_column(String(1024))

    # ----------------------------------------------------- AI-derived fields
    category: Mapped[ComplaintCategory] = mapped_column(
        _enum_column(ComplaintCategory, 24), default=ComplaintCategory.OTHER, nullable=False
    )
    priority: Mapped[ComplaintPriority] = mapped_column(
        _enum_column(ComplaintPriority, 16), default=ComplaintPriority.MEDIUM, nullable=False
    )
    ai_summary: Mapped[str | None] = mapped_column(Text)

    #: The complete AI record: every prediction, its confidence, the runner-up
    #: candidates, which engine ran, the model version and the processing time.
    #: Denormalised copies live in `category` / `priority` / `ai_summary` for querying.
    ai_output: Mapped[dict[str, Any] | None] = mapped_column(JSONColumn)

    #: True once an administrator overrides an AI prediction. Lets us measure how often
    #: humans disagree with the model — the honest way to report real-world accuracy.
    ai_overridden: Mapped[bool] = mapped_column(default=False, nullable=False)

    # ----------------------------------------------------------- management
    status: Mapped[ComplaintStatus] = mapped_column(
        _enum_column(ComplaintStatus, 24), default=ComplaintStatus.OPEN, nullable=False, index=True
    )
    assigned_department_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("departments.id", ondelete="SET NULL"), index=True
    )
    assigned_department: Mapped[Department | None] = relationship(
        back_populates="complaints", lazy="selectin"
    )
    resolution_note: Mapped[str | None] = mapped_column(Text)
    resolved_at: Mapped[datetime | None] = mapped_column(TimestampColumn, index=True)

    # ---------------------------------------------------------- duplicates
    duplicate_of_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("complaints.id", ondelete="SET NULL")
    )
    duplicate_of: Mapped[Complaint | None] = relationship(
        remote_side="Complaint.id", lazy="noload"
    )

    events: Mapped[list[StatusEvent]] = relationship(
        back_populates="complaint",
        cascade="all, delete-orphan",
        order_by="StatusEvent.created_at",
        lazy="selectin",
    )

    # ------------------------------------------------------------ behaviour
    @property
    def resolution_hours(self) -> float | None:
        """Hours between submission and resolution, or ``None`` if still open."""
        if self.resolved_at is None:
            return None
        return (self.resolved_at - self.created_at).total_seconds() / 3600.0

    @property
    def age_hours(self) -> float:
        """Hours since submission (frozen at resolution time for closed complaints)."""
        end = self.resolved_at or utcnow()
        return (end - self.created_at).total_seconds() / 3600.0

    @property
    def is_overdue(self) -> bool:
        """True when an unresolved complaint has passed its priority's SLA target."""
        if self.status.is_terminal:
            return False
        return self.age_hours > self.priority.target_resolution_hours

    @property
    def ai_confidence(self) -> float | None:
        if not self.ai_output:
            return None
        value = self.ai_output.get("category_confidence")
        return float(value) if value is not None else None

    def apply_ai_result(self, result: dict[str, Any]) -> None:
        """Absorb an ``AIResult`` payload.

        Called on creation and whenever an administrator re-runs analysis. It never
        overwrites a human override — that is the point of ``ai_overridden``.
        """
        self.ai_output = result
        self.ai_summary = result.get("summary")
        if not self.ai_overridden:
            if category := result.get("category"):
                self.category = ComplaintCategory(category)
            if priority := result.get("priority"):
                self.priority = ComplaintPriority(priority)

    def mark_resolved(self, note: str | None = None) -> None:
        self.status = ComplaintStatus.RESOLVED
        self.resolved_at = utcnow()
        if note:
            self.resolution_note = note

    def reopen(self) -> None:
        self.resolved_at = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Complaint {self.reference_code} {self.category}/{self.priority} {self.status}>"
