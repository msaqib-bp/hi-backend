"""Domain exceptions and the handlers that turn them into HTTP responses.

Services raise semantic errors (``ComplaintNotFoundError``) and know nothing about HTTP.
The handlers registered in ``register_exception_handlers`` map them onto status codes, so
the same service layer could sit behind a CLI or a worker without change.

Every error response shares one envelope so the frontend can handle failures uniformly:

    {"error": {"type": "not_found", "message": "...", "detail": {...}}}
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from app.core.logging import current_request_id, get_logger

log = get_logger(__name__)

#: Starlette renamed this constant (ENTITY -> CONTENT) and deprecated the old spelling.
#: Using the literal keeps us working across both, since Render resolves its own version.
HTTP_422_UNPROCESSABLE = 422


# --------------------------------------------------------------------- domain errors
class CivicServiceError(Exception):
    """Base class for every expected, domain-level failure."""

    status_code: int = status.HTTP_400_BAD_REQUEST
    error_type: str = "error"

    def __init__(self, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}


class NotFoundError(CivicServiceError):
    status_code = status.HTTP_404_NOT_FOUND
    error_type = "not_found"


class ComplaintNotFoundError(NotFoundError):
    def __init__(self, identifier: str) -> None:
        super().__init__(
            f"No complaint found for '{identifier}'.",
            {"identifier": identifier},
        )


class DepartmentNotFoundError(NotFoundError):
    def __init__(self, identifier: str) -> None:
        super().__init__(f"No department found for '{identifier}'.", {"identifier": identifier})


class ValidationError(CivicServiceError):
    status_code = HTTP_422_UNPROCESSABLE
    error_type = "validation_error"


class InvalidStatusTransitionError(CivicServiceError):
    """Guards the complaint lifecycle: you cannot resolve something already rejected."""

    status_code = status.HTTP_409_CONFLICT
    error_type = "invalid_transition"

    def __init__(self, current: str, requested: str, allowed: list[str]) -> None:
        super().__init__(
            f"Cannot move a complaint from '{current}' to '{requested}'.",
            {"current_status": current, "requested_status": requested, "allowed": allowed},
        )


class AuthenticationError(CivicServiceError):
    status_code = status.HTTP_401_UNAUTHORIZED
    error_type = "authentication_error"


class PermissionError_(CivicServiceError):
    status_code = status.HTTP_403_FORBIDDEN
    error_type = "permission_denied"


class AIServiceError(CivicServiceError):
    """Raised only by the low-level analyzers; ``AIPipeline`` catches these."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_type = "ai_unavailable"


# ------------------------------------------------------------------------- handlers
def _envelope(error_type: str, message: str, detail: Any = None) -> dict[str, Any]:
    return {
        "error": {
            "type": error_type,
            "message": message,
            "detail": detail or {},
            "request_id": current_request_id(),
        }
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(CivicServiceError)
    async def _domain_error(_: Request, exc: CivicServiceError) -> JSONResponse:
        log.warning("domain_error", type=exc.error_type, message=exc.message, detail=exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.error_type, exc.message, exc.detail),
        )

    @app.exception_handler(RequestValidationError)
    async def _request_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Flatten Pydantic's error list into something a form can display per-field.
        fields = {
            ".".join(str(part) for part in err["loc"][1:]) or "body": err["msg"]
            for err in exc.errors()
        }
        return JSONResponse(
            status_code=HTTP_422_UNPROCESSABLE,
            content=_envelope("validation_error", "The submitted data is not valid.", fields),
        )

    @app.exception_handler(SQLAlchemyError)
    async def _database_error(_: Request, exc: SQLAlchemyError) -> JSONResponse:
        log.error("database_error", error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_envelope(
                "database_error",
                "The service could not reach its database. Please try again shortly.",
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_error", error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope("internal_error", "An unexpected error occurred."),
        )
