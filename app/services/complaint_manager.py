"""``ComplaintManager`` — the application's business logic.

Everything that happens to a complaint goes through this class: intake, AI analysis,
department routing, duplicate checking, lifecycle transitions and the audit trail. The
API layer above it does HTTP and nothing else; this layer knows nothing about HTTP.

The AI is *inside* the workflow, not beside it — a complaint cannot be created without
passing through the analyzer, which is what the spec means by "AI output must become
part of the application workflow, not a separate demonstration".
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Select, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import (
    ComplaintNotFoundError,
    DepartmentNotFoundError,
    InvalidStatusTransitionError,
)
from app.core.logging import get_logger
from app.models.base import utcnow
from app.models.complaint import Complaint, generate_reference_code
from app.models.department import Department
from app.models.enums import (
    CATEGORY_TO_DEPARTMENT_SLUG,
    STATUS_TRANSITIONS,
    ComplaintCategory,
    ComplaintPriority,
    ComplaintStatus,
)
from app.models.status_event import StatusEvent
from app.schemas.complaint import ComplaintCreate, ComplaintUpdate
from app.services.ai.duplicates import DuplicateDetector
from app.services.ai.pipeline import AIPipeline, get_ai_pipeline
from app.services.notifications import NotificationManager

log = get_logger(__name__)


class ComplaintManager:
    """Owns the complaint lifecycle.

    Dependencies are injected rather than constructed internally, so tests can supply a
    stub analyzer and assert on behaviour without touching a model file.
    """

    def __init__(
        self,
        session: AsyncSession,
        ai_pipeline: AIPipeline | None = None,
        notifier: NotificationManager | None = None,
    ) -> None:
        self._session = session
        self._ai = ai_pipeline or get_ai_pipeline()
        self._notifier = notifier or NotificationManager()
        self._duplicates = DuplicateDetector(session)

    # --------------------------------------------------------------------- create
    async def create(
        self, payload: ComplaintCreate, *, check_duplicates: bool = True
    ) -> tuple[Complaint, list[dict[str, Any]]]:
        """Intake a citizen complaint: analyse it, route it, store it.

        Returns the complaint and any likely duplicates. Duplicates are *reported*, not
        acted on — merging automatically would silently discard a genuine report when
        the similarity heuristic is wrong.
        """
        ai_result = await self._ai.analyze(payload.description, payload.location)

        complaint = Complaint(
            reference_code=await self._unique_reference_code(),
            description=payload.description,
            location=payload.location,
            latitude=payload.latitude,
            longitude=payload.longitude,
            reporter_name=payload.reporter_name,
            reporter_contact=payload.reporter_contact,
            image_url=payload.image_url,
            status=ComplaintStatus.OPEN,
        )
        complaint.apply_ai_result(ai_result.to_dict())

        department = await self._department_for_category(complaint.category)
        if department is not None:
            complaint.assigned_department = department
            complaint.status = ComplaintStatus.ASSIGNED

        self._session.add(complaint)
        await self._session.flush()  # assigns the PK so the event can reference it

        self._session.add(
            StatusEvent(
                complaint_id=complaint.id,
                from_status=None,
                to_status=complaint.status,
                actor="system",
                note=(
                    f"Classified as {complaint.category.label} / "
                    f"{complaint.priority.label} by the {ai_result.engine.value} engine "
                    f"({ai_result.category_confidence:.0%} confidence)."
                ),
            )
        )

        duplicates: list[dict[str, Any]] = []
        if check_duplicates:
            duplicates = await self._duplicates.find_duplicates(
                payload.description, payload.location, exclude_id=complaint.id
            )

        self._notifier.notify_new_complaint(complaint)
        await self._session.flush()
        await self._session.refresh(complaint)

        log.info(
            "complaint_created",
            reference=complaint.reference_code,
            category=complaint.category.value,
            priority=complaint.priority.value,
            duplicates=len(duplicates),
        )
        return complaint, duplicates

    async def _unique_reference_code(self, attempts: int = 6) -> str:
        """Generate a tracking code, retrying on the (rare) collision.

        31^6 ≈ 887M combinations, so a collision is very unlikely — but "very unlikely"
        becomes "happened in the demo" often enough to be worth six lines.
        """
        for _ in range(attempts):
            code = generate_reference_code()
            existing = await self._session.execute(
                select(Complaint.id).where(Complaint.reference_code == code)
            )
            if existing.scalar_one_or_none() is None:
                return code
        # Astronomically improbable; fall back to a UUID fragment rather than looping.
        return f"CIV-{uuid.uuid4().hex[:8].upper()}"

    async def _department_for_category(
        self, category: ComplaintCategory
    ) -> Department | None:
        slug = CATEGORY_TO_DEPARTMENT_SLUG.get(category)
        if not slug:
            return None
        result = await self._session.execute(
            select(Department).where(Department.slug == slug, Department.is_active.is_(True))
        )
        return result.scalar_one_or_none()

    # ---------------------------------------------------------------------- read
    async def get_by_id(self, complaint_id: uuid.UUID) -> Complaint:
        result = await self._session.execute(select(Complaint).where(Complaint.id == complaint_id))
        complaint = result.scalar_one_or_none()
        if complaint is None:
            raise ComplaintNotFoundError(str(complaint_id))
        return complaint

    async def get_by_reference(self, reference_code: str) -> Complaint:
        """Public tracking lookup. Case-insensitive — people type it from memory."""
        normalised = reference_code.strip().upper()
        result = await self._session.execute(
            select(Complaint).where(func.upper(Complaint.reference_code) == normalised)
        )
        complaint = result.scalar_one_or_none()
        if complaint is None:
            raise ComplaintNotFoundError(reference_code)
        return complaint

    async def search(
        self,
        *,
        status: ComplaintStatus | None = None,
        category: ComplaintCategory | None = None,
        priority: ComplaintPriority | None = None,
        department_id: uuid.UUID | None = None,
        location: str | None = None,
        query: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        overdue_only: bool = False,
        page: int = 1,
        page_size: int = 20,
        sort: str = "-created_at",
    ) -> tuple[list[Complaint], int]:
        """Filtered, paginated complaint search. Returns ``(items, total)``."""
        statement = select(Complaint)
        statement = self._apply_filters(
            statement,
            status=status,
            category=category,
            priority=priority,
            department_id=department_id,
            location=location,
            query=query,
            date_from=date_from,
            date_to=date_to,
        )

        # Count before pagination, over the same filters.
        count_statement = select(func.count()).select_from(statement.subquery())
        total = (await self._session.execute(count_statement)).scalar_one()

        statement = statement.order_by(self._sort_clause(sort))
        page = max(page, 1)
        page_size = min(max(page_size, 1), 100)
        statement = statement.offset((page - 1) * page_size).limit(page_size)

        result = await self._session.execute(statement)
        items = list(result.scalars().all())

        # ``is_overdue`` compares against the current clock, so it cannot be expressed
        # as a portable SQL predicate across SQLite and Postgres. Filtering in Python
        # is acceptable because it only ever runs over one page of results.
        if overdue_only:
            items = [complaint for complaint in items if complaint.is_overdue]

        return items, total

    def _apply_filters(
        self,
        statement: Select,
        *,
        status: ComplaintStatus | None,
        category: ComplaintCategory | None,
        priority: ComplaintPriority | None,
        department_id: uuid.UUID | None,
        location: str | None,
        query: str | None,
        date_from: datetime | None,
        date_to: datetime | None,
    ) -> Select:
        if status is not None:
            statement = statement.where(Complaint.status == status)
        if category is not None:
            statement = statement.where(Complaint.category == category)
        if priority is not None:
            statement = statement.where(Complaint.priority == priority)
        if department_id is not None:
            statement = statement.where(Complaint.assigned_department_id == department_id)
        if location:
            statement = statement.where(Complaint.location.ilike(f"%{location}%"))
        if query:
            pattern = f"%{query.strip()}%"
            statement = statement.where(
                or_(
                    Complaint.description.ilike(pattern),
                    Complaint.reference_code.ilike(pattern),
                    Complaint.location.ilike(pattern),
                    Complaint.ai_summary.ilike(pattern),
                )
            )
        if date_from is not None:
            statement = statement.where(Complaint.created_at >= date_from)
        if date_to is not None:
            statement = statement.where(Complaint.created_at <= date_to)
        return statement

    @staticmethod
    def _sort_clause(sort: str):
        column_name = sort.lstrip("-")
        allowed = {
            "created_at": Complaint.created_at,
            "updated_at": Complaint.updated_at,
            "priority": Complaint.priority,
            "status": Complaint.status,
            "category": Complaint.category,
        }
        column = allowed.get(column_name, Complaint.created_at)
        return column.desc() if sort.startswith("-") else column.asc()

    # -------------------------------------------------------------------- update
    async def update(
        self, complaint_id: uuid.UUID, payload: ComplaintUpdate, *, actor: str = "admin"
    ) -> Complaint:
        """Apply an administrator's changes, enforcing the lifecycle rules."""
        complaint = await self.get_by_id(complaint_id)
        previous_status = complaint.status
        status_changed = False

        if payload.assigned_department_id is not None:
            department = await self._session.get(Department, payload.assigned_department_id)
            if department is None:
                raise DepartmentNotFoundError(str(payload.assigned_department_id))
            complaint.assigned_department = department

        # A human correcting the AI is a signal worth keeping: `ai_overridden` feeds the
        # override rate on the dashboard, which is our honest real-world accuracy metric.
        if payload.category is not None and payload.category != complaint.category:
            complaint.category = payload.category
            complaint.ai_overridden = True
        if payload.priority is not None and payload.priority != complaint.priority:
            complaint.priority = payload.priority
            complaint.ai_overridden = True

        if payload.resolution_note is not None:
            complaint.resolution_note = payload.resolution_note

        if payload.status is not None and payload.status != complaint.status:
            self._assert_transition_allowed(complaint.status, payload.status)
            complaint.status = payload.status
            status_changed = True

            if payload.status is ComplaintStatus.RESOLVED:
                complaint.resolved_at = utcnow()
            elif previous_status is ComplaintStatus.RESOLVED:
                # Reopening: clear the timestamp so resolution-time statistics do not
                # count a complaint that is open again as still resolved.
                complaint.reopen()

        if status_changed:
            self._session.add(
                StatusEvent(
                    complaint_id=complaint.id,
                    from_status=previous_status,
                    to_status=complaint.status,
                    actor=actor,
                    note=payload.note,
                )
            )
            self._notifier.notify_status_change(complaint, note=payload.note)

        await self._session.flush()
        await self._session.refresh(complaint)
        log.info(
            "complaint_updated",
            reference=complaint.reference_code,
            status=complaint.status.value,
            overridden=complaint.ai_overridden,
        )
        return complaint

    @staticmethod
    def _assert_transition_allowed(
        current: ComplaintStatus, requested: ComplaintStatus
    ) -> None:
        allowed = STATUS_TRANSITIONS.get(current, set())
        if requested not in allowed:
            raise InvalidStatusTransitionError(
                current.value, requested.value, sorted(status.value for status in allowed)
            )

    # ----------------------------------------------------------------- reanalyze
    async def reanalyze(self, complaint_id: uuid.UUID) -> tuple[Complaint, dict, dict]:
        """Re-run the AI over an existing complaint.

        Used after a model retrain, or when an admin wants a second opinion. Human
        overrides are preserved — ``apply_ai_result`` will not overwrite a label a
        person deliberately corrected.
        """
        complaint = await self.get_by_id(complaint_id)
        previous = dict(complaint.ai_output or {})

        result = await self._ai.analyze(complaint.description, complaint.location)
        complaint.apply_ai_result(result.to_dict())

        await self._session.flush()
        await self._session.refresh(complaint)

        current = complaint.ai_output or {}
        changed = (
            previous.get("category") != current.get("category")
            or previous.get("priority") != current.get("priority")
        )
        return complaint, previous, {"changed": changed, **current}

    async def mark_duplicate(
        self, complaint_id: uuid.UUID, original_id: uuid.UUID, *, actor: str = "admin"
    ) -> Complaint:
        """Link a complaint to the original it duplicates and close it."""
        complaint = await self.get_by_id(complaint_id)
        original = await self.get_by_id(original_id)

        if complaint.id == original.id:
            from app.core.exceptions import ValidationError

            raise ValidationError("A complaint cannot be a duplicate of itself.")

        previous_status = complaint.status
        complaint.duplicate_of_id = original.id
        complaint.status = ComplaintStatus.REJECTED
        complaint.resolution_note = (
            f"Closed as a duplicate of {original.reference_code}."
        )

        self._session.add(
            StatusEvent(
                complaint_id=complaint.id,
                from_status=previous_status,
                to_status=complaint.status,
                actor=actor,
                note=f"Marked as duplicate of {original.reference_code}.",
            )
        )
        self._notifier.notify_status_change(complaint, note=complaint.resolution_note)

        await self._session.flush()
        await self._session.refresh(complaint)
        return complaint

    async def find_duplicates_for(self, complaint_id: uuid.UUID) -> list[dict[str, Any]]:
        complaint = await self.get_by_id(complaint_id)
        return await self._duplicates.find_duplicates(
            complaint.description, complaint.location, exclude_id=complaint.id
        )
