"""Tests for connection-string normalisation.

Every hosted Postgres vendor hands out a slightly different URL, and the app has to
accept whichever one gets pasted into ``DATABASE_URL`` without anyone editing it by
hand. Two independent things can go wrong:

1. **The scheme.** SQLAlchemy needs ``postgresql+asyncpg://``; Render emits the legacy
   ``postgres://`` and Neon emits ``postgresql://``.

2. **The query parameters.** SQLAlchemy's asyncpg dialect forwards every query
   parameter to ``asyncpg.connect()`` as a keyword argument, and asyncpg accepts none
   of libpq's spellings. Neon's string ends ``?sslmode=require&channel_binding=require``,
   which raises ``TypeError: connect() got an unexpected keyword argument 'sslmode'`` —
   and it raises it on the *first connection*, i.e. inside ``alembic upgrade head``
   during deploy, long after the config parsed cleanly.

The second failure is the one worth pinning hardest, because its symptom (deploy dies
at the migration step) points nowhere near its cause (a query parameter). The final test
class asserts against ``asyncpg.connect``'s real signature rather than a hand-written
list, so it keeps telling the truth when the driver is upgraded.
"""

from __future__ import annotations

import inspect

import asyncpg
import pytest
from sqlalchemy.dialects.postgresql.asyncpg import PGDialect_asyncpg
from sqlalchemy.engine.url import make_url

from app.core.config import Settings, normalise_database_url
from app.db.session import _uses_transaction_pooler

#: The exact strings these providers put on their dashboards.
NEON_DIRECT = (
    "postgresql://civic_owner:npg_secret@ep-holy-frost-a4x9.us-east-2.aws.neon.tech"
    "/civic?sslmode=require&channel_binding=require"
)
NEON_POOLED = (
    "postgresql://civic_owner:npg_secret@ep-holy-frost-a4x9-pooler.us-east-2.aws"
    ".neon.tech/civic?sslmode=require"
)
RENDER_LEGACY = "postgres://civic:pw@dpg-abc123-a.oregon-postgres.render.com/civic"
SUPABASE_POOLED = "postgresql://postgres.abcd:pw@pooler.supabase.com:6543/postgres?sslmode=require"
SQLITE_DEFAULT = "sqlite+aiosqlite:////srv/civic.db"


class TestSchemeNormalisation:
    def test_render_legacy_scheme_gets_the_async_driver(self) -> None:
        assert normalise_database_url(RENDER_LEGACY).startswith("postgresql+asyncpg://")

    def test_standard_scheme_gets_the_async_driver(self) -> None:
        assert normalise_database_url(NEON_DIRECT).startswith("postgresql+asyncpg://")

    def test_already_correct_url_is_left_alone(self) -> None:
        url = "postgresql+asyncpg://u:p@host/db"
        assert normalise_database_url(url) == url

    def test_sqlite_is_untouched(self) -> None:
        assert normalise_database_url(SQLITE_DEFAULT) == SQLITE_DEFAULT

    def test_only_the_scheme_is_rewritten(self) -> None:
        """A password containing the literal text ``postgres://`` must survive."""
        url = "postgres://user:postgres://weird@host/db"
        assert normalise_database_url(url) == (
            "postgresql+asyncpg://user:postgres://weird@host/db"
        )


class TestLibpqParameters:
    def test_sslmode_is_translated_not_discarded(self) -> None:
        """Dropping it would silently downgrade the connection to unencrypted."""
        result = normalise_database_url(NEON_DIRECT)
        assert "sslmode=" not in result
        assert "ssl=require" in result

    def test_ssl_verification_level_is_preserved(self) -> None:
        url = "postgresql://u:p@host/db?sslmode=verify-full"
        assert "ssl=verify-full" in normalise_database_url(url)

    def test_channel_binding_is_dropped(self) -> None:
        """asyncpg has no equivalent, so the only options are drop it or crash."""
        assert "channel_binding" not in normalise_database_url(NEON_DIRECT)

    def test_unknown_parameters_are_preserved(self) -> None:
        """The drop list is explicit; anything else may be a real asyncpg argument."""
        url = "postgresql://u:p@host/db?sslmode=require&statement_cache_size=0"
        assert "statement_cache_size=0" in normalise_database_url(url)

    def test_credentials_survive_the_rewrite(self) -> None:
        result = normalise_database_url(NEON_DIRECT)
        assert "civic_owner:npg_secret@" in result
        assert result.endswith("/civic?ssl=require")

    def test_port_survives_the_rewrite(self) -> None:
        assert ":6543/postgres" in normalise_database_url(SUPABASE_POOLED)

    def test_settings_applies_the_same_normalisation(self) -> None:
        """Alembic and the app both read ``settings.DATABASE_URL``, so the fix has to
        live on the field, not at one call site."""
        settings = Settings(DATABASE_URL=NEON_DIRECT)
        assert settings.DATABASE_URL.startswith("postgresql+asyncpg://")
        assert "sslmode" not in settings.DATABASE_URL


class TestPoolerDetection:
    """Transaction pooling needs asyncpg's prepared-statement cache switched off."""

    @pytest.mark.parametrize("url", [NEON_POOLED, SUPABASE_POOLED])
    def test_pooled_endpoints_are_recognised(self, url: str) -> None:
        assert _uses_transaction_pooler(normalise_database_url(url)) is True

    @pytest.mark.parametrize("url", [NEON_DIRECT, RENDER_LEGACY])
    def test_direct_endpoints_are_not(self, url: str) -> None:
        assert _uses_transaction_pooler(normalise_database_url(url)) is False


class TestEveryKwargIsAcceptedByTheDriver:
    """The regression that motivated this module, asserted against the real driver.

    ``create_connect_args`` returns precisely what SQLAlchemy will splat into
    ``asyncpg.connect()``. Comparing it to that function's live signature catches both
    a parameter we forgot to strip and a future asyncpg release that stops accepting
    one we currently pass.
    """

    @pytest.mark.parametrize(
        "url",
        [NEON_DIRECT, NEON_POOLED, RENDER_LEGACY, SUPABASE_POOLED],
        ids=["neon-direct", "neon-pooled", "render-legacy", "supabase-pooled"],
    )
    def test_no_kwarg_would_be_rejected(self, url: str) -> None:
        _, kwargs = PGDialect_asyncpg().create_connect_args(
            make_url(normalise_database_url(url))
        )
        # SQLAlchemy's own adapter pops this one before asyncpg ever sees it.
        kwargs.pop("prepared_statement_cache_size", None)

        accepted = set(inspect.signature(asyncpg.connect).parameters)
        rejected = sorted(set(kwargs) - accepted)
        assert not rejected, f"asyncpg.connect() would raise TypeError on: {rejected}"

    def test_the_unfixed_url_really_would_have_failed(self) -> None:
        """Guards the guard: if this ever passes, the test above proves nothing."""
        naive = NEON_DIRECT.replace("postgresql://", "postgresql+asyncpg://", 1)
        _, kwargs = PGDialect_asyncpg().create_connect_args(make_url(naive))

        accepted = set(inspect.signature(asyncpg.connect).parameters)
        assert "sslmode" in set(kwargs) - accepted
