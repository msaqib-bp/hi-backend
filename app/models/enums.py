"""Domain enumerations.

Values are lowercase snake_case because they travel over JSON to the frontend and get
used directly as query-string filters. Labels are the human-readable forms rendered in
the UI and used when the LLM is prompted.
"""

from __future__ import annotations

from enum import StrEnum


class ComplaintCategory(StrEnum):
    """The service areas a complaint can be routed to (spec §4)."""

    ROAD = "road"
    WATER = "water"
    WASTE = "waste"
    ELECTRICITY = "electricity"
    DRAINAGE = "drainage"
    SAFETY = "safety"
    OTHER = "other"

    @property
    def label(self) -> str:
        return {
            ComplaintCategory.ROAD: "Road & Footpath",
            ComplaintCategory.WATER: "Water Supply",
            ComplaintCategory.WASTE: "Waste & Sanitation",
            ComplaintCategory.ELECTRICITY: "Electricity & Streetlights",
            ComplaintCategory.DRAINAGE: "Drainage & Sewerage",
            ComplaintCategory.SAFETY: "Public Safety",
            ComplaintCategory.OTHER: "Other Services",
        }[self]


class ComplaintPriority(StrEnum):
    """Urgency estimated by the AI, overridable by an administrator."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def label(self) -> str:
        return self.value.capitalize()

    @property
    def rank(self) -> int:
        """Numeric weight — used for sorting queues and for statistics."""
        return {
            ComplaintPriority.LOW: 1,
            ComplaintPriority.MEDIUM: 2,
            ComplaintPriority.HIGH: 3,
            ComplaintPriority.CRITICAL: 4,
        }[self]

    @property
    def target_resolution_hours(self) -> int:
        """Service-level target, used to flag breaches on the dashboard."""
        return {
            ComplaintPriority.CRITICAL: 6,
            ComplaintPriority.HIGH: 24,
            ComplaintPriority.MEDIUM: 72,
            ComplaintPriority.LOW: 168,
        }[self]


class ComplaintStatus(StrEnum):
    """Lifecycle states. Transitions are enforced by ``ComplaintManager``."""

    OPEN = "open"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    REJECTED = "rejected"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()

    @property
    def is_terminal(self) -> bool:
        return self in (ComplaintStatus.RESOLVED, ComplaintStatus.REJECTED)


#: Allowed state machine. A complaint may only move along these edges.
STATUS_TRANSITIONS: dict[ComplaintStatus, set[ComplaintStatus]] = {
    ComplaintStatus.OPEN: {
        ComplaintStatus.ASSIGNED,
        ComplaintStatus.IN_PROGRESS,
        ComplaintStatus.REJECTED,
    },
    ComplaintStatus.ASSIGNED: {
        ComplaintStatus.IN_PROGRESS,
        ComplaintStatus.RESOLVED,
        ComplaintStatus.REJECTED,
        ComplaintStatus.OPEN,
    },
    ComplaintStatus.IN_PROGRESS: {
        ComplaintStatus.RESOLVED,
        ComplaintStatus.REJECTED,
        ComplaintStatus.ASSIGNED,
    },
    # Terminal states can be reopened, which is why they are not empty sets.
    ComplaintStatus.RESOLVED: {ComplaintStatus.IN_PROGRESS},
    ComplaintStatus.REJECTED: {ComplaintStatus.OPEN},
}


class UserRole(StrEnum):
    ADMIN = "admin"
    STAFF = "staff"

    @property
    def label(self) -> str:
        return self.value.capitalize()


class AIEngine(StrEnum):
    """Which engine produced a given AI result — surfaced in the UI for transparency."""

    ML = "ml"
    LLM = "llm"
    HYBRID = "hybrid"
    FALLBACK = "fallback"  # heuristics only; both engines were unavailable


#: Default routing table: which department owns which category.
#: The AI predicts the category; this map turns that into an owning team.
CATEGORY_TO_DEPARTMENT_SLUG: dict[ComplaintCategory, str] = {
    ComplaintCategory.ROAD: "public-works",
    ComplaintCategory.WATER: "water-supply",
    ComplaintCategory.WASTE: "sanitation",
    ComplaintCategory.ELECTRICITY: "electrical",
    ComplaintCategory.DRAINAGE: "drainage-sewerage",
    ComplaintCategory.SAFETY: "public-safety",
    ComplaintCategory.OTHER: "general-administration",
}
