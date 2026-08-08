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
import pathlib
import ssl

import asyncpg
import pytest
from sqlalchemy.dialects.postgresql.asyncpg import PGDialect_asyncpg
from sqlalchemy.engine.url import make_url

from app.core.config import Settings, normalise_database_url
from app.db.engine_options import (
    certificate_verifying_context,
    connect_args_for,
    uses_transaction_pooler,
)
from app.db.session import DatabaseManager

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
        assert uses_transaction_pooler(normalise_database_url(url)) is True

    @pytest.mark.parametrize("url", [NEON_DIRECT, RENDER_LEGACY])
    def test_direct_endpoints_are_not(self, url: str) -> None:
        assert uses_transaction_pooler(normalise_database_url(url)) is False


class TestCertificateVerification:
    """``verify-ca``/``verify-full`` must not need a certificate file on disk.

    asyncpg resolves those modes against ``~/.postgresql/root.crt`` and raises when it is
    missing, which on a container it always is. Left alone, the strongest setting is the
    one that cannot boot — and the fallback, ``require``, connects with
    ``verify_mode = CERT_NONE``, so it encrypts without authenticating anything.
    """

    @pytest.mark.parametrize("mode", ["disable", "prefer", "require"])
    def test_non_verifying_modes_are_left_to_asyncpg(self, mode: str) -> None:
        url = f"postgresql+asyncpg://u:p@host/db?ssl={mode}"
        assert certificate_verifying_context(url) is None

    def test_verify_ca_checks_the_chain_but_not_the_hostname(self) -> None:
        context = certificate_verifying_context(
            "postgresql+asyncpg://u:p@host/db?ssl=verify-ca"
        )
        assert context is not None
        assert context.verify_mode is ssl.CERT_REQUIRED
        assert context.check_hostname is False

    def test_verify_full_checks_the_hostname_too(self) -> None:
        context = certificate_verifying_context(
            "postgresql+asyncpg://u:p@host/db?ssl=verify-full"
        )
        assert context is not None
        assert context.verify_mode is ssl.CERT_REQUIRED
        assert context.check_hostname is True

    def test_the_context_trusts_a_public_ca(self) -> None:
        """Managed providers use public certificates — Neon's is Let's Encrypt."""
        context = certificate_verifying_context(
            "postgresql+asyncpg://u:p@host/db?ssl=verify-full"
        )
        assert context is not None
        assert context.get_ca_certs(), "no trust store was loaded"

    def test_certifi_covers_an_image_with_no_system_trust_store(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A slim base image can omit ``ca-certificates`` entirely.

        Verification against an empty trust store rejects *every* certificate, which
        would turn this hardening into an outage — so an empty store has to fall back
        rather than proceed.
        """
        empty = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        assert not empty.get_ca_certs()
        monkeypatch.setattr(ssl, "create_default_context", lambda *a, **kw: empty)

        context = certificate_verifying_context(
            "postgresql+asyncpg://u:p@host/db?ssl=verify-full"
        )
        assert context is not None
        assert context.get_ca_certs(), "fell back to nothing — every cert would be rejected"

    def test_the_context_overrides_the_url(self) -> None:
        """The URL still says ``ssl=verify-full``, and asyncpg would act on that string
        by looking for ``root.crt``. The context must reach ``connect_args``, which
        SQLAlchemy merges over the query parameters."""
        url = normalise_database_url(NEON_DIRECT.replace("sslmode=require", "sslmode=verify-full"))
        manager = DatabaseManager(url)
        assert isinstance(manager.connect_args.get("ssl"), ssl.SSLContext)

    def test_require_adds_no_override(self) -> None:
        manager = DatabaseManager(normalise_database_url(NEON_DIRECT))
        assert "ssl" not in manager.connect_args

    def test_pooled_and_verifying_settings_combine(self) -> None:
        """A pooled endpoint with verification on needs *both* adjustments, not one.

        These are set on different branches, so it would be easy for one to overwrite
        the other's ``connect_args``.
        """
        url = normalise_database_url(NEON_POOLED.replace("sslmode=require", "sslmode=verify-full"))
        manager = DatabaseManager(url)
        assert manager.is_pooled is True
        assert isinstance(manager.connect_args.get("ssl"), ssl.SSLContext)
        assert manager.connect_args["statement_cache_size"] == 0


class TestMigrationsAndAppConnectIdentically:
    """The application and Alembic open separate engines, and they must agree.

    Sharing ``settings.DATABASE_URL`` is not enough on its own: an ``SSLContext`` is an
    object, so it cannot ride inside a connection string. When only ``DatabaseManager``
    supplied one, a ``sslmode=verify-full`` URL let the app connect while
    ``alembic upgrade head`` failed on asyncpg's missing ``root.crt`` — during deploy,
    before the app ever started, which is the worst place to discover it.
    """

    @pytest.mark.parametrize(
        "url", [NEON_DIRECT, NEON_POOLED, RENDER_LEGACY, SQLITE_DEFAULT]
    )
    def test_the_manager_adds_nothing_of_its_own(self, url: str) -> None:
        """Whatever the manager uses must come from the shared builder, so that Alembic
        — which calls the builder directly — receives exactly the same thing."""
        normalised = normalise_database_url(url)
        assert DatabaseManager(normalised).connect_args == connect_args_for(normalised)

    def test_alembic_env_wires_the_shared_builder(self) -> None:
        """Read as a contract on the file: the regression this guards is someone
        constructing Alembic's engine without ``connect_args``, which cannot be caught
        behaviourally without running a real migration against a TLS database."""
        env = (
            pathlib.Path(__file__).resolve().parent.parent / "alembic" / "env.py"
        ).read_text()
        assert "connect_args_for" in env, "alembic/env.py no longer uses the shared builder"
        assert "connect_args=connect_args_for(" in env, (
            "alembic/env.py imports the builder but does not pass it to its engine"
        )

    def test_sqlite_needs_no_overrides_either_way(self) -> None:
        """The local default must stay a plain engine — none of this applies to it."""
        assert connect_args_for(SQLITE_DEFAULT) == {}


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
