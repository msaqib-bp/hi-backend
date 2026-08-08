"""Alembic environment, configured for the async engine.

The database URL comes from ``Settings`` rather than ``alembic.ini`` so migrations use
exactly the same connection string as the application — including Render's
``postgres://`` → ``postgresql+asyncpg://`` normalisation.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

import app.models  # noqa: F401,E402  (side-effect import)
from alembic import context
from app.core.config import settings

# Importing the models package registers every table on Base.metadata, which is what
# autogenerate diffs against.
from app.models import Base  # noqa: F401

config = context.config
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting — useful for reviewing a migration."""
    context.configure(
        url=settings.DATABASE_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_item=render_item,
    )
    with context.begin_transaction():
        context.run_migrations()


def render_item(type_: str, obj, autogen_context) -> str | bool:  # noqa: ANN001
    """Render custom column types as their plain SQLAlchemy equivalent.

    ``UTCDateTime`` is a ``TypeDecorator`` living in application code. Autogenerate
    would emit ``app.models.base.UTCDateTime(...)`` into the migration, which both fails
    (the module is not imported there) and couples a permanent migration to code that
    may be renamed later. Migrations should stand alone, so emit ``sa.DateTime`` — the
    underlying column type is identical.
    """
    import sqlalchemy as sa

    from app.models.base import UTCDateTime

    if type_ != "type":
        return False

    if isinstance(obj, UTCDateTime):
        return "sa.DateTime(timezone=True)"

    # The JSON/JSONB variant renders by default as
    # ``postgresql.JSONB(astext_type=Text())`` without importing ``Text``, which is a
    # NameError at migration time. Emit the equivalent without that argument.
    if isinstance(obj, sa.JSON) and getattr(obj, "_variant_mapping", None):
        autogen_context.imports.add("from sqlalchemy.dialects import postgresql")
        return "sa.JSON().with_variant(postgresql.JSONB(), 'postgresql')"

    return False  # fall through to Alembic's default rendering


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        render_item=render_item,
        # SQLite cannot ALTER most columns; batch mode rewrites the table instead.
        render_as_batch=connection.dialect.name == "sqlite",
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
