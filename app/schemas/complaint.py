"""Request and response contracts for complaints."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.enums import ComplaintCategory, ComplaintPriority, ComplaintStatus


# ------------------------------------------------------------------------- inbound
class ComplaintCreate(BaseModel):
    """What a citizen submits. Only description and location are required — asking for
    more up front measurably reduces the number of reports people finish."""

    description: str = Field(
        ...,
        min_length=15,
        max_length=5000,
        description="What is wrong, in the citizen's own words. This is the AI's input.",
        examples=["There is a large water leak near the main road and traffic is becoming difficult."],
    )
    location: str = Field(..., min_length=3, max_length=255, examples=["MG Road, Ward 12"])
    latitude: float | None = Field(None, ge=-90, le=90)
    longitude: float | None = Field(None, ge=-180, le=180)
    reporter_name: str | None = Field(None, max_length=160)
    reporter_contact: str | None = Field(None, max_length=160)
    image_url: str | None = Field(None, max_length=1024)

    @field_validator("description", "location")
    @classmethod
    def _strip(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("This field cannot be blank.")
        return cleaned

    @field_validator("description")
    @classmethod
    def _reject_placeholder(cls, value: str) -> str:
        # Guards against "asdasdasd" submissions that would poison the statistics.
        if len(set(value.lower().replace(" ", ""))) < 5:
            raise ValueError("Please describe the problem in a few real words.")
        return value


class ComplaintUpdate(BaseModel):
    """Administrator edits. Every field optional — this is a PATCH."""

    status: ComplaintStatus | None = None
    category: ComplaintCategory | None = None
    priority: ComplaintPriority | None = None
    assigned_department_id: uuid.UUID | None = None
    resolution_note: str | None = Field(None, max_length=2000)
    note: str | None = Field(
        None, max_length=500, description="Optional note recorded on the status timeline."
    )


# ------------------------------------------------------------------------ outbound
class DepartmentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    slug: str


class StatusEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    from_status: ComplaintStatus | None
    to_status: ComplaintStatus
    note: str | None
    actor: str
    created_at: datetime


class AIOutputOut(BaseModel):
    """The AI record, surfaced verbatim so the UI can explain the prediction."""

    category: ComplaintCategory | None = None
    category_confidence: float | None = None
    category_alternatives: list[dict[str, Any]] = Field(default_factory=list)
    priority: ComplaintPriority | None = None
    priority_confidence: float | None = None
    priority_alternatives: list[dict[str, Any]] = Field(default_factory=list)
    summary: str | None = None
    recommended_department: str | None = None
    keywords: list[str] = Field(default_factory=list)
    engine: str | None = None
    model_version: str | None = None
    processing_ms: float | None = None
    notes: list[str] = Field(default_factory=list)

    # ``model_`` is a protected Pydantic namespace by default; we want the field name.
    model_config = ConfigDict(protected_namespaces=())


class ComplaintOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    reference_code: str
    description: str
    location: str
    latitude: float | None
    longitude: float | None
    reporter_name: str | None
    reporter_contact: str | None
    image_url: str | None

    category: ComplaintCategory
    priority: ComplaintPriority
    status: ComplaintStatus
    ai_summary: str | None
    ai_output: AIOutputOut | None
    ai_overridden: bool

    assigned_department: DepartmentSummary | None
    resolution_note: str | None
    duplicate_of_id: uuid.UUID | None

    created_at: datetime
    updated_at: datetime
    resolved_at: datetime | None

    # Computed properties from the entity — exposed so the UI never recalculates them.
    age_hours: float
    resolution_hours: float | None
    is_overdue: bool

    events: list[StatusEventOut] = Field(default_factory=list)


class ComplaintListItem(BaseModel):
    """Trimmed shape for tables — avoids shipping full descriptions and timelines."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    reference_code: str
    description: str
    location: str
    category: ComplaintCategory
    priority: ComplaintPriority
    status: ComplaintStatus
    ai_summary: str | None
    assigned_department: DepartmentSummary | None
    created_at: datetime
    resolved_at: datetime | None
    age_hours: float
    is_overdue: bool


class DuplicateCandidate(BaseModel):
    id: uuid.UUID
    reference_code: str
    description: str
    similarity: float
    status: ComplaintStatus
    created_at: datetime


class ComplaintCreateResponse(BaseModel):
    """Returned straight after submission — this is what powers the citizen's
    'here is what the AI understood' card."""

    complaint: ComplaintOut
    possible_duplicates: list[DuplicateCandidate] = Field(default_factory=list)
    message: str


class PaginatedComplaints(BaseModel):
    items: list[ComplaintListItem]
    total: int
    page: int
    page_size: int
    pages: int
