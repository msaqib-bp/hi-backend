"""FastAPI application factory and entry point.

Run locally::

    uvicorn app.main:app --reload

Interactive API docs are served at ``/docs`` — that surface is deliberate. It gives the
demo a live, explorable view of every endpoint without building anything, and doubles as
the API reference for the submission.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger, new_request_id
from app.db.seed import run_seed
from app.db.session import db_manager
from app.services.ai.pipeline import get_ai_pipeline

configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown.

    Three things happen before the first request: the schema is ensured, the AI models
    are loaded (~1s, and doing it here rather than lazily means the first citizen to
    submit a complaint does not pay for it), and demo data is seeded if the database is
    empty.
    """
    log.info(
        "application_starting",
        environment=settings.ENVIRONMENT,
        version=settings.APP_VERSION,
        # Logged because a CORS rejection is invisible from the server's side: the
        # request runs and returns 200, and only the browser discards the response.
        # Without this line the allowed list can only be discovered by guessing.
        cors_origins=settings.cors_origin_list,
    )
    if settings.cors_is_probably_misconfigured:
        log.warning(
            "cors_origins_not_configured",
            allowed=settings.cors_origin_list,
            hint=(
                "Only localhost is allowed, so every browser request from the deployed "
                "frontend will be blocked. Set CORS_ORIGINS to the site's origin — "
                "scheme and host only, e.g. https://your-app.vercel.app (no trailing "
                "slash, no path)."
            ),
        )

    # On SQLite (local dev and tests) create the tables directly, because nothing else
    # will. On Postgres, skip it: `alembic upgrade head` runs in the deploy step and owns
    # the schema.
    #
    # The reason is correctness, not speed. Creating tables behind Alembic's back leaves
    # no `alembic_version` row, so the next `upgrade head` tries to create a schema that
    # already exists and fails — on a database that looks perfectly healthy.
    #
    # It does also take a couple of network round trips off the startup path, but that is
    # a rounding error next to the platform's own cold start; do not mistake this for a
    # fix for slow boots.
    if db_manager.is_sqlite:
        await db_manager.create_all()
    else:
        log.info("schema_owned_by_migrations", hint="alembic upgrade head runs at deploy")

    pipeline = get_ai_pipeline()
    log.info(
        "ai_pipeline_ready",
        engine=pipeline.active_engine,
        ml=pipeline.ml.available,
        llm=pipeline.llm.available,
    )
    if not pipeline.ml.available:
        log.warning(
            "ml_models_unavailable",
            hint="Run `python -m app.ml.train` to enable the trained classifiers.",
        )

    try:
        async with db_manager.session_scope() as session:
            await run_seed(session)
    except Exception as exc:  # pragma: no cover - seeding must never block startup
        log.error("seeding_failed", error=str(exc), exc_info=True)

    yield

    await db_manager.dispose()
    log.info("application_stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        lifespan=lifespan,
        description=(
            "Civic complaint intake, AI triage and analytics.\n\n"
            "Citizens submit complaints in free text; the AI classifies the category, "
            "estimates priority, routes to a department and writes an actionable summary. "
            "Administrators manage the queue and read the statistics.\n\n"
            "**Public endpoints:** submit a complaint, track by reference code, all "
            "analytics. **Admin endpoints:** everything that lists or mutates complaints."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Tag every request with an id and log its outcome and duration."""
        request_id = new_request_id()
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Handlers registered below convert domain errors; anything reaching here
            # is a genuine bug, so log it with the request id before re-raising.
            log.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                exc_info=True,
            )
            raise

        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        log.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            ms=round(duration_ms, 1),
        )
        return response

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # GET *and* HEAD. FastAPI does not add HEAD alongside GET, and Render's platform
    # checker — like most uptime monitors — probes with HEAD, which was answering 405.
    # That matters here beyond tidiness: an uptime pinger is the usual way to stop a
    # free instance sleeping before a demo, and a 405 makes it report the service down.
    @app.api_route("/", methods=["GET", "HEAD"], tags=["meta"], summary="Service metadata")
    async def root() -> JSONResponse:
        pipeline = get_ai_pipeline()
        return JSONResponse(
            {
                "service": settings.APP_NAME,
                "version": settings.APP_VERSION,
                "status": "operational",
                "docs": "/docs",
                "api": settings.API_V1_PREFIX,
                "ai_engine": pipeline.active_engine,
            }
        )

    @app.api_route(
        "/health", methods=["GET", "HEAD"], tags=["meta"], summary="Health check"
    )
    async def health() -> JSONResponse:
        """Liveness probe for Render. Verifies the database round-trips."""
        from sqlalchemy import text

        database_ok = True
        try:
            async with db_manager.session_scope() as session:
                await session.execute(text("SELECT 1"))
        except Exception as exc:
            database_ok = False
            log.error("health_check_database_failed", error=str(exc))

        pipeline = get_ai_pipeline()
        payload = {
            "status": "healthy" if database_ok else "degraded",
            "database": "up" if database_ok else "down",
            "ai": {
                "engine": pipeline.active_engine,
                "ml_loaded": pipeline.ml.available,
                "llm_configured": pipeline.llm.available,
            },
            "version": settings.APP_VERSION,
        }
        return JSONResponse(payload, status_code=200 if database_ok else 503)

    return app


app = create_app()
