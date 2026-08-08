"""Test fixtures.

Each test runs against a fresh in-memory SQLite database with demo seeding disabled, so
tests are isolated and fast. The AI pipeline is stubbed by default — the goal is to test
*our* logic deterministically, not to re-measure scikit-learn on every run. The tests
that genuinely exercise the real models live in ``test_ai.py`` and opt in explicitly.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

# Must be set before app.core.config is imported anywhere.
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("SEED_DEMO_DATA", "false")
# 32+ bytes, matching the production minimum, so tests exercise a realistic key length.
os.environ.setdefault("SECRET_KEY", "test-secret-key-that-is-long-enough-for-hs256-signing")
os.environ.setdefault("ENVIRONMENT", "test")

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.seed import seed_admin_user, seed_departments  # noqa: E402
from app.models.base import Base  # noqa: E402
from app.models.enums import AIEngine, ComplaintCategory, ComplaintPriority  # noqa: E402
from app.services.ai.base import AIAnalyzer, AIResult, Prediction  # noqa: E402


class StubAnalyzer(AIAnalyzer):
    """Deterministic analyzer so service tests assert on behaviour, not model output."""

    name = "stub"

    def __init__(
        self,
        category: ComplaintCategory = ComplaintCategory.WATER,
        priority: ComplaintPriority = ComplaintPriority.HIGH,
        *,
        available: bool = True,
        raises: bool = False,
    ) -> None:
        self._category = category
        self._priority = priority
        self._available = available
        self._raises = raises
        self.calls = 0

    @property
    def available(self) -> bool:
        return self._available

    async def analyze(self, description: str, location: str | None = None) -> AIResult:
        self.calls += 1
        if self._raises:
            from app.core.exceptions import AIServiceError

            raise AIServiceError("stub failure")
        return AIResult(
            category=self._category,
            priority=self._priority,
            summary=f"Stub summary for: {description[:40]}",
            engine=AIEngine.ML,
            model_version="stub-1.0",
            category_confidence=0.9,
            priority_confidence=0.8,
            category_alternatives=[Prediction("other", 0.05)],
            keywords=["stub"],
        )


@pytest.fixture
async def engine():
    """A fresh in-memory database per test."""
    # StaticPool keeps every connection pointed at the same in-memory database;
    # without it each connection would get its own empty one.
    from sqlalchemy.pool import StaticPool

    test_engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield test_engine
    await test_engine.dispose()


@pytest.fixture
async def session(engine) -> AsyncIterator[AsyncSession]:
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as test_session:
        yield test_session


@pytest.fixture
async def seeded_session(session: AsyncSession) -> AsyncSession:
    """Session with departments and the admin user in place."""
    await seed_departments(session)
    await seed_admin_user(session)
    await session.commit()
    return session


@pytest.fixture
def stub_analyzer() -> StubAnalyzer:
    return StubAnalyzer()


@pytest.fixture
async def client(engine, seeded_session) -> AsyncIterator[AsyncClient]:
    """HTTP client wired to the app with the test database and a stubbed AI engine."""
    from app.api.deps import get_pipeline
    from app.db.session import get_session
    from app.main import create_app
    from app.services.ai.pipeline import AIPipeline

    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_session() -> AsyncIterator[AsyncSession]:
        async with factory() as test_session:
            try:
                yield test_session
                await test_session.commit()
            except Exception:
                await test_session.rollback()
                raise

    pipeline = AIPipeline(
        ml_analyzer=StubAnalyzer(),  # type: ignore[arg-type]
        llm_analyzer=StubAnalyzer(available=False),  # type: ignore[arg-type]
        use_llm_for_summary=False,
    )

    app = create_app()
    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_pipeline] = lambda: pipeline

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client


@pytest.fixture
async def admin_token(client: AsyncClient) -> str:
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "admin@civic.gov", "password": "admin123"},
    )
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


@pytest.fixture
def auth_headers(admin_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {admin_token}"}
