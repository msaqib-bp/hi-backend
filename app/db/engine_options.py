"""Driver arguments a connection URL needs beyond what SQLAlchemy derives from it.

This module exists because there are **two** independent places that open a connection:
the application (`DatabaseManager`) and Alembic (`alembic/env.py`, which builds its own
engine with `async_engine_from_config`). They already share the URL — both read
`settings.DATABASE_URL` — but sharing the URL is not enough, because some of what asyncpg
needs cannot be expressed *in* a URL. An `ssl.SSLContext` is an object, not a string.

Keeping that logic here, rather than inside `DatabaseManager`, means migrations and the
app cannot connect on different terms. The failure that motivated it was exactly that
asymmetry: with `sslmode=verify-full`, the app connected and `alembic upgrade head`
died — during deploy, before the app ever started.

Deliberately free of side effects, so `alembic/env.py` can import it without constructing
the application's engine.
"""

from __future__ import annotations

import ssl
from urllib.parse import parse_qsl, urlsplit

import certifi

#: ``ssl`` values that ask for the certificate to actually be checked. Both spellings
#: appear because ``urlencode`` preserves the hyphen while asyncpg's enum uses an
#: underscore.
_VERIFYING_SSL_MODES = frozenset({"verify-ca", "verify_ca", "verify-full", "verify_full"})


def certificate_verifying_context(database_url: str) -> ssl.SSLContext | None:
    """Supply a trust store when the URL asks for certificate verification.

    asyncpg resolves ``verify-ca`` and ``verify-full`` against libpq's default
    certificate path, ``~/.postgresql/root.crt``, and raises if it is absent — it never
    falls back to the system trust store::

        ClientConfigurationError: root certificate file
        "/home/…/.postgresql/root.crt" does not exist

    That file exists on almost no container, so without this the *strongest* setting is
    the one that fails to boot, leaving ``sslmode=require`` as the only working option —
    and asyncpg implements ``require`` with ``verify_mode = CERT_NONE``, which encrypts
    the connection but authenticates nothing. An attacker able to intercept the route
    could present any certificate at all.

    Every managed provider uses a publicly-trusted certificate (Neon's is Let's Encrypt),
    so a public trust store is precisely what is needed to close that gap.

    Returns ``None`` unless verification was requested, leaving asyncpg's own handling of
    ``disable``/``prefer``/``require`` exactly as it was.

    Not covered: a database behind a *private* CA, whose ``sslrootcert`` this does not
    read. That fails loudly with a verification error rather than connecting insecurely,
    and no major hosted provider needs it.
    """
    requested = dict(parse_qsl(urlsplit(database_url).query)).get("ssl", "").lower()
    if requested not in _VERIFYING_SSL_MODES:
        return None

    context = ssl.create_default_context()
    if not context.get_ca_certs():
        # A slim base image can ship without the ``ca-certificates`` package, leaving
        # OpenSSL's default verify paths empty. Verification would then reject every
        # certificate, turning a security improvement into an outage. certifi carries
        # the same Mozilla root set and is already installed.
        context.load_verify_locations(cafile=certifi.where())

    context.verify_mode = ssl.CERT_REQUIRED
    # verify-ca checks the chain but not the name; verify-full checks both.
    context.check_hostname = requested.startswith(("verify-full", "verify_full"))
    return context


def uses_transaction_pooler(database_url: str) -> bool:
    """True for a connection string pointing at a PgBouncer transaction pooler.

    Neon and Supabase both expose two endpoints for the same database and show the
    pooled one first, so it is the string people actually copy. Its hostname carries a
    ``-pooler`` marker (Neon) or a ``pooler.`` prefix (Supabase).
    """
    host = urlsplit(database_url).hostname or ""
    return "-pooler." in host or host.startswith("pooler.")


def connect_args_for(database_url: str) -> dict:
    """Everything asyncpg needs for this URL that the URL itself cannot carry.

    Both connection paths call this, so migrations and the application always connect on
    identical terms.
    """
    args: dict = {}

    context = certificate_verifying_context(database_url)
    if context is not None:
        # Overrides the ``ssl=verify-…`` string still in the URL: SQLAlchemy merges
        # ``connect_args`` over the query parameters, so asyncpg receives the ready
        # context and never reaches its ``root.crt`` lookup.
        args["ssl"] = context

    if not database_url.startswith("sqlite") and uses_transaction_pooler(database_url):
        # Under transaction pooling each statement may land on a different backend.
        # asyncpg caches server-side prepared statements by name, so the second query
        # hits a connection that never saw the PREPARE:
        #   InvalidSQLStatementNameError: prepared statement "__asyncpg_stmt_1__"
        #   does not exist
        # Both caches have to go — one is asyncpg's, the other SQLAlchemy's. The cost is
        # re-planning each statement; the alternative is a database that fails
        # intermittently under load, which is far worse to debug live.
        args |= {"statement_cache_size": 0, "prepared_statement_cache_size": 0}

    return args
