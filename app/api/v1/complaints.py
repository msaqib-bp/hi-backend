"""Complaint endpoints.

Submission and tracking are **public** — the spec's user journey has a citizen file a
report and follow it by reference code with no account. Everything that mutates or lists
complaints in bulk requires an administrator token.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from math import ceil
from typing import Annotated

from fastapi import APIRouter, Query, status

from app.api.deps import AdminUser, ManagerDep
from app.models.enums import ComplaintCategory, ComplaintPriority, ComplaintStatus
from app.schemas.complaint import (
    ComplaintCreate,
    ComplaintCreateResponse,
    ComplaintListItem,
    ComplaintOut,
    ComplaintUpdate,
    DuplicateCandidate,
    PaginatedComplaints,
)

router = APIRouter(prefix="/complaints", tags=["complaints"])


@router.post(
    "",
    response_model=ComplaintCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a complaint (public)",
)
async def create_complaint(
    payload: ComplaintCreate, manager: ManagerDep
) -> ComplaintCreateResponse:
    """Submit a civic complaint.

    The AI runs inline: the response already contains the predicted category, priority,
    routed department and dispatch summary, which is what the citizen sees confirmed
    back to them. Any likely duplicates are reported but never merged automatically.
    """
    complaint, duplicates = await manager.create(payload)

    message = (
        f"Complaint {complaint.reference_code} received and routed to "
        f"{complaint.assigned_department.name if complaint.assigned_department else 'the relevant team'}."
    )
    if duplicates:
        message += (
            f" Note: {len(duplicates)} similar complaint(s) are already open — yours has "
            "still been recorded."
        )

    return ComplaintCreateResponse(
        complaint=ComplaintOut.model_validate(complaint),
        possible_duplicates=[DuplicateCandidate(**item) for item in duplicates],
        message=message,
    )


@router.get(
    "/track/{reference_code}",
    response_model=ComplaintOut,
    summary="Track a complaint by reference code (public)",
)
async def track_complaint(reference_code: str, manager: ManagerDep) -> ComplaintOut:
    """Look up a complaint by its ``CIV-XXXXXX`` code. No authentication required."""
    complaint = await manager.get_by_reference(reference_code)
    return ComplaintOut.model_validate(complaint)


@router.get("", response_model=PaginatedComplaints, summary="List and filter complaints")
async def list_complaints(
    manager: ManagerDep,
    _: AdminUser,
    status_filter: Annotated[ComplaintStatus | None, Query(alias="status")] = None,
    category: ComplaintCategory | None = None,
    priority: ComplaintPriority | None = None,
    department_id: uuid.UUID | None = None,
    location: Annotated[str | None, Query(max_length=255)] = None,
    q: Annotated[str | None, Query(max_length=255, description="Free-text search")] = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    overdue_only: bool = False,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    sort: Annotated[str, Query(pattern=r"^-?(created_at|updated_at|priority|status|category)$")] = "-created_at",
) -> PaginatedComplaints:
    """Search complaints across every filter dimension the spec asks for:
    category, priority, status, date, location and department."""
    items, total = await manager.search(
        status=status_filter,
        category=category,
        priority=priority,
        department_id=department_id,
        location=location,
        query=q,
        date_from=date_from,
        date_to=date_to,
        overdue_only=overdue_only,
        page=page,
        page_size=page_size,
        sort=sort,
    )
    return PaginatedComplaints(
        items=[ComplaintListItem.model_validate(item) for item in items],
        total=total,
        page=page,
        page_size=page_size,
        pages=ceil(total / page_size) if page_size else 0,
    )


@router.get("/{complaint_id}", response_model=ComplaintOut, summary="Get one complaint")
async def get_complaint(
    complaint_id: uuid.UUID, manager: ManagerDep, _: AdminUser
) -> ComplaintOut:
    complaint = await manager.get_by_id(complaint_id)
    return ComplaintOut.model_validate(complaint)


@router.patch("/{complaint_id}", response_model=ComplaintOut, summary="Update a complaint")
async def update_complaint(
    complaint_id: uuid.UUID,
    payload: ComplaintUpdate,
    manager: ManagerDep,
    admin: AdminUser,
) -> ComplaintOut:
    """Change status, reassign, or override an AI classification.

    Overriding a category or priority sets ``ai_overridden``, which feeds the override
    rate on the dashboard — our honest measure of how often the model is wrong on real
    complaints, as opposed to on a held-out test split.
    """
    complaint = await manager.update(complaint_id, payload, actor=admin.full_name)
    return ComplaintOut.model_validate(complaint)


@router.get(
    "/{complaint_id}/duplicates",
    response_model=list[DuplicateCandidate],
    summary="Find likely duplicates of a complaint",
)
async def complaint_duplicates(
    complaint_id: uuid.UUID, manager: ManagerDep, _: AdminUser
) -> list[DuplicateCandidate]:
    candidates = await manager.find_duplicates_for(complaint_id)
    return [DuplicateCandidate(**item) for item in candidates]


@router.post(
    "/{complaint_id}/duplicate-of/{original_id}",
    response_model=ComplaintOut,
    summary="Close a complaint as a duplicate",
)
async def mark_duplicate(
    complaint_id: uuid.UUID,
    original_id: uuid.UUID,
    manager: ManagerDep,
    admin: AdminUser,
) -> ComplaintOut:
    complaint = await manager.mark_duplicate(complaint_id, original_id, actor=admin.full_name)
    return ComplaintOut.model_validate(complaint)
