"""Tests for settings that are only exercised in a deployed environment.

These are the values nobody sets locally and everybody has to set in production,
which is precisely why they are worth pinning: a mistake surfaces for the first time
on a live deployment, in front of whoever is looking at it.
"""

from __future__ import annotations

from app.core.config import Settings


class TestCorsOriginParsing:
    """A CORS rejection is invisible server-side, so the parsing has to be forgiving.

    ``CORSMiddleware`` matches the ``Origin`` header by exact string. A browser sends
    scheme, host and port only — never a trailing slash, never a path — but the value
    people paste is copied from the address bar, which has both. When it fails to match,
    the request still succeeds and returns 200; only the browser discards the response,
    for want of an ``access-control-allow-origin`` header. Nothing in the server logs
    says why.
    """

    def test_a_trailing_slash_is_tolerated(self) -> None:
        """The single most likely way to get this wrong: pasting the address bar."""
        settings = Settings(CORS_ORIGINS="https://hi-frontend-blush.vercel.app/")
        assert settings.cors_origin_list == ["https://hi-frontend-blush.vercel.app"]

    def test_multiple_origins_split_and_strip(self) -> None:
        settings = Settings(
            CORS_ORIGINS="https://app.vercel.app/ , http://localhost:3000 ,"
        )
        assert settings.cors_origin_list == [
            "https://app.vercel.app",
            "http://localhost:3000",
        ]

    def test_duplicates_collapse(self) -> None:
        """The slash-and-not-slash pair is the same origin once normalised."""
        settings = Settings(CORS_ORIGINS="https://a.app,https://a.app/")
        assert settings.cors_origin_list == ["https://a.app"]

    def test_a_port_is_preserved(self) -> None:
        """Origins compare including the port — stripping it would break local dev."""
        settings = Settings(CORS_ORIGINS="http://localhost:3000/")
        assert settings.cors_origin_list == ["http://localhost:3000"]

    def test_production_with_only_localhost_is_flagged(self) -> None:
        settings = Settings(ENVIRONMENT="production", SECRET_KEY="x" * 40)
        assert settings.cors_is_probably_misconfigured is True

    def test_production_with_a_real_origin_is_not(self) -> None:
        settings = Settings(
            ENVIRONMENT="production",
            SECRET_KEY="x" * 40,
            CORS_ORIGINS="https://hi-frontend-blush.vercel.app/",
        )
        assert settings.cors_is_probably_misconfigured is False

    def test_development_is_never_flagged(self) -> None:
        """Localhost-only is exactly right locally; warning there would be noise."""
        assert Settings(ENVIRONMENT="development").cors_is_probably_misconfigured is False
