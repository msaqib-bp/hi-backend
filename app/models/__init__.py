"""ORM models.

Imported as a package so that Alembic's autogenerate and ``Base.metadata.create_all``
both see every table.
"""

from app.models.base import Base
from app.models.complaint import Complaint, generate_reference_code
from app.models.department import Department
from app.models.enums import (
    CATEGORY_TO_DEPARTMENT_SLUG,
    STATUS_TRANSITIONS,
    AIEngine,
    ComplaintCategory,
    ComplaintPriority,
    ComplaintStatus,
    UserRole,
)
from app.models.status_event import StatusEvent
from app.models.user import User

__all__ = [
    "CATEGORY_TO_DEPARTMENT_SLUG",
    "STATUS_TRANSITIONS",
    "AIEngine",
    "Base",
    "Complaint",
    "ComplaintCategory",
    "ComplaintPriority",
    "ComplaintStatus",
    "Department",
    "StatusEvent",
    "User",
    "UserRole",
    "generate_reference_code",
]
