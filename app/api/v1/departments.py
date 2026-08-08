"""Department endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import SessionDep
from app.models.department import Department
from app.schemas.complaint import DepartmentSummary

router = APIRouter(prefix="/departments", tags=["departments"])


@router.get("", response_model=list[DepartmentSummary], summary="List service departments")
async def list_departments(session: SessionDep) -> list[DepartmentSummary]:
    """Public — the admin UI needs this to populate the reassignment dropdown."""
    result = await session.execute(
        select(Department).where(Department.is_active.is_(True)).order_by(Department.name)
    )
    return [DepartmentSummary.model_validate(row) for row in result.scalars().all()]
