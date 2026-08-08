"""Database engine, session factory and the ``DatabaseManager`` facade.

``DatabaseManager`` exists because the spec (Batch 5) asks for a class that owns database
responsibilities rather than scattering engine handling across modules. It holds the
engine lifecycle, hands out sessions, and knows how to create the schema for local
development and tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings
from app.core.logging import get_logger
from app.db.engine_options import connect_args_for, uses_transaction_pooler
from app.models.base import Base

log = get_logger(__name__)


class DatabaseManager:
    """Owns the async engine and session factory for one process."""

    def __init__(self, database_url: str, *, echo: bool = False) -> None:
        self._database_url = database_url
        self._is_sqlite = database_url.startswith("sqlite")
        self._is_pooled = not self._is_sqlite and uses_transaction_pooler(database_url)

        # SQLite ignores pool sizing; hosted Postgres free tiers have a low connection
        # cap, so keep the pool small and recycle before the server drops idle
        # connections.
        engine_kwargs: dict = {"echo": echo, "future": True, "pool_pre_ping": True}
        if not self._is_sqlite:
            engine_kwargs |= {"pool_size": 5, "max_overflow": 5, "pool_recycle": 300}

        # Built by the shared module Alembic also uses, so migrations and the
        # application always connect on identical terms.
        connect_args = connect_args_for(database_url)
        self._connect_args = connect_args
        if connect_args:
            engine_kwargs["connect_args"] = connect_args

        self._engine: AsyncEngine = create_async_engine(database_url, **engine_kwargs)
        self._session_factory = async_sessionmaker(
            self._engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
        )

    # ---------------------------------------------------------------- access
    @property
    def engine(self) -> AsyncEngine:
        return self._engine

    @property
    def is_sqlite(self) -> bool:
        return self._is_sqlite

    @property
    def is_pooled(self) -> bool:
        """True when connected through a PgBouncer transaction pooler."""
        return self._is_pooled

    @property
    def connect_args(self) -> dict:
        """Driver arguments this manager decided to override, for tests and diagnostics."""
        return dict(self._connect_args)

    def session(self) -> AsyncSession:
        """Return a new session. The caller owns commit/rollback/close."""
        return self._session_factory()

    @asynccontextmanager
    async def session_scope(self) -> AsyncIterator[AsyncSession]:
        """Transactional scope: commits on success, rolls back on any exception."""
        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    # ------------------------------------------------------------- lifecycle
    async def create_all(self) -> None:
        """Create every table. Used for tests and SQLite dev; production uses Alembic."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        log.info("database_schema_ready", sqlite=self._is_sqlite)

    async def drop_all(self) -> None:  # pragma: no cover - test helper
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    async def dispose(self) -> None:
        await self._engine.dispose()
        log.info("database_engine_disposed")


#: Process-wide instance used by the application.
db_manager = DatabaseManager(settings.DATABASE_URL, echo=settings.DB_ECHO)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a request-scoped session.

    Commits when the handler returns cleanly so route code does not have to remember to;
    rolls back on any exception raised out of the handler.
    """
    session = db_manager.session()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
