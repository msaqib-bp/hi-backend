"""Shared FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import AuthenticationError, PermissionError_
from app.core.security import decode_access_token
from app.db.session import get_session
from app.models.user import User
from app.services.ai.pipeline import AIPipeline, get_ai_pipeline
from app.services.complaint_manager import ComplaintManager
from app.services.statistics import StatisticsService

SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_pipeline() -> AIPipeline:
    return get_ai_pipeline()


PipelineDep = Annotated[AIPipeline, Depends(get_pipeline)]


def get_complaint_manager(session: SessionDep, pipeline: PipelineDep) -> ComplaintManager:
    return ComplaintManager(session, pipeline)


def get_statistics_service(session: SessionDep) -> StatisticsService:
    return StatisticsService(session)


ManagerDep = Annotated[ComplaintManager, Depends(get_complaint_manager)]
StatsDep = Annotated[StatisticsService, Depends(get_statistics_service)]


async def get_current_user(
    session: SessionDep,
    authorization: Annotated[str | None, Header()] = None,
) -> User:
    """Resolve the signed-in administrator from the Bearer token."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthenticationError("Sign in to access this resource.")

    token = authorization.split(" ", 1)[1].strip()
    payload = decode_access_token(token)
    subject = payload.get("sub")
    if not subject:
        raise AuthenticationError("Malformed authentication token.")

    result = await session.execute(select(User).where(User.email == subject))
    user = result.scalar_one_or_none()
    if user is None:
        raise AuthenticationError("This account no longer exists.")
    if not user.is_active:
        raise PermissionError_("This account has been deactivated.")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def require_admin(user: CurrentUser) -> User:
    if not user.is_admin:
        raise PermissionError_("This action requires an administrator account.")
    return user


AdminUser = Annotated[User, Depends(require_admin)]
