"""Authentication endpoints for the administrator dashboard."""

from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import func, select

from app.api.deps import CurrentUser, SessionDep
from app.core.config import settings
from app.core.exceptions import AuthenticationError
from app.core.logging import get_logger
from app.core.security import create_access_token
from app.models.user import User
from app.schemas.auth import LoginRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])
log = get_logger(__name__)


@router.post("/login", response_model=TokenResponse, summary="Sign in")
async def login(payload: LoginRequest, session: SessionDep) -> TokenResponse:
    """Exchange credentials for a JWT."""
    result = await session.execute(
        select(User).where(func.lower(User.email) == payload.email.lower())
    )
    user = result.scalar_one_or_none()

    # One message for both "no such user" and "wrong password" — distinguishing them
    # would let an attacker enumerate valid accounts.
    if user is None or not user.check_password(payload.password):
        log.info("login_failed", email=payload.email)
        raise AuthenticationError("Incorrect email or password.")

    if not user.is_active:
        raise AuthenticationError("This account has been deactivated.")

    token = create_access_token(user.email, {"role": user.role.value, "name": user.full_name})
    log.info("login_succeeded", email=user.email, role=user.role.value)

    return TokenResponse(
        access_token=token,
        expires_in_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES,
        user=UserOut.model_validate(user),
    )


@router.get("/me", response_model=UserOut, summary="Current signed-in user")
async def me(user: CurrentUser) -> UserOut:
    return UserOut.model_validate(user)
