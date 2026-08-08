"""Application configuration.

Everything the app needs to run is read from the environment (or a local ``.env``)
through a single validated ``Settings`` object. Nothing else in the codebase reads
``os.environ`` directly, so the full configuration surface is visible in one file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent  # -> backend/app
PROJECT_ROOT = BASE_DIR.parent  # -> backend/


class Settings(BaseSettings):
    """Validated application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ app
    APP_NAME: str = "AI Smart Civic Services"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"  # development | production
    DEBUG: bool = True
    API_V1_PREFIX: str = "/api/v1"

    # ----------------------------------------------------------------- http
    # Comma-separated list of allowed browser origins (the Vercel URL in prod).
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    # ------------------------------------------------------------- database
    # Local default is a SQLite file; Render injects a Postgres URL.
    DATABASE_URL: str = f"sqlite+aiosqlite:///{PROJECT_ROOT / 'civic.db'}"
    DB_ECHO: bool = False

    # ------------------------------------------------------------- security
    SECRET_KEY: str = "dev-only-secret-change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 12  # 12 hours

    # Bootstrap admin, created on first startup if no users exist.
    ADMIN_EMAIL: str = "admin@civic.gov"
    ADMIN_PASSWORD: str = "admin123"
    ADMIN_NAME: str = "Municipal Administrator"

    # -------------------------------------------------------------------- ai
    # The language model is entirely OPTIONAL. With no key at all, the local
    # scikit-learn models handle every mandatory capability and the app is fully
    # functional — an LLM only improves the dispatch summary and enables the
    # natural-language assistant.
    #
    # Two providers are supported. Set whichever key you have:
    #   ANTHROPIC_API_KEY  -> Claude
    #   DEEPSEEK_API_KEY   -> DeepSeek (or any OpenAI-compatible endpoint)
    # If both are set, LLM_PROVIDER decides; "auto" prefers Anthropic.
    LLM_PROVIDER: str = "auto"  # auto | anthropic | deepseek | openai_compatible | none

    ANTHROPIC_API_KEY: str | None = None
    ANTHROPIC_MODEL: str = "claude-haiku-4-5"

    DEEPSEEK_API_KEY: str | None = None
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"

    # Generic overrides for any other OpenAI-compatible vendor (Groq, Together,
    # OpenRouter, a local vLLM/Ollama server). Leave unset to use the DeepSeek values.
    OPENAI_COMPATIBLE_API_KEY: str | None = None
    OPENAI_COMPATIBLE_MODEL: str | None = None
    OPENAI_COMPATIBLE_BASE_URL: str | None = None

    LLM_TIMEOUT_SECONDS: float = 20.0
    LLM_MAX_TOKENS: int = 1024

    ML_ARTIFACT_DIR: Path = BASE_DIR / "ml" / "artifacts"
    MODEL_VERSION: str = "1.0.0"

    # Cosine-similarity threshold above which two complaints are considered
    # possible duplicates. Tuned on the evaluation set; see reports/.
    DUPLICATE_SIMILARITY_THRESHOLD: float = 0.55
    DUPLICATE_LOOKBACK_DAYS: int = 30

    # ------------------------------------------------------------------ seed
    SEED_DEMO_DATA: bool = True
    SEED_COMPLAINT_COUNT: int = 180

    # ------------------------------------------------------------- computed
    @field_validator("DATABASE_URL")
    @classmethod
    def _normalise_database_url(cls, value: str) -> str:
        """Render hands out ``postgres://`` URLs; SQLAlchemy needs an async driver."""
        if value.startswith("postgres://"):
            value = value.replace("postgres://", "postgresql+asyncpg://", 1)
        elif value.startswith("postgresql://"):
            value = value.replace("postgresql://", "postgresql+asyncpg://", 1)
        return value

    @field_validator("SECRET_KEY")
    @classmethod
    def _reject_weak_secret_in_production(cls, value: str, info) -> str:  # noqa: ANN001
        """A short or default signing key makes every admin token forgeable.

        Only enforced in production so local development and tests stay frictionless —
        but a deployed instance must not run on the shipped default.
        """
        environment = (info.data or {}).get("ENVIRONMENT", "development")
        if str(environment).lower() != "production":
            return value

        if value.startswith("dev-only-") or len(value.encode()) < 32:
            raise ValueError(
                "SECRET_KEY must be a private value of at least 32 bytes in production. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(48))\""
            )
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    # ------------------------------------------------------ llm provider choice
    @staticmethod
    def _clean(value: str | None) -> str:
        return (value or "").strip()

    @property
    def anthropic_configured(self) -> bool:
        return bool(self._clean(self.ANTHROPIC_API_KEY))

    @property
    def openai_compatible_configured(self) -> bool:
        return bool(
            self._clean(self.OPENAI_COMPATIBLE_API_KEY) or self._clean(self.DEEPSEEK_API_KEY)
        )

    @property
    def active_llm_provider(self) -> str:
        """Which LLM provider to use: ``anthropic``, ``openai_compatible`` or ``none``.

        ``auto`` (the default) picks whichever key is present, preferring Anthropic when
        both are — it uses a strict tool schema, so its output cannot fail validation the
        way a JSON-mode response can. An explicit setting is honoured even if that
        provider's key is missing, so a misconfiguration surfaces as "not configured"
        rather than silently running on the other vendor's bill.
        """
        choice = self._clean(self.LLM_PROVIDER).lower() or "auto"

        if choice in {"deepseek", "openai", "openai_compatible"}:
            return "openai_compatible" if self.openai_compatible_configured else "none"
        if choice == "anthropic":
            return "anthropic" if self.anthropic_configured else "none"
        if choice == "none":
            return "none"

        # auto
        if self.anthropic_configured:
            return "anthropic"
        if self.openai_compatible_configured:
            return "openai_compatible"
        return "none"

    @property
    def llm_enabled(self) -> bool:
        """The LLM engine is opt-in: no key means the ML engine runs alone."""
        return self.active_llm_provider != "none"

    # --- resolved settings for the OpenAI-compatible provider -------------------
    # The generic OPENAI_COMPATIBLE_* variables win when set; otherwise the DeepSeek
    # defaults apply, so a user with only DEEPSEEK_API_KEY needs no other configuration.
    @property
    def LLM_API_KEY(self) -> str:  # noqa: N802 - reads as a setting at the call site
        return self._clean(self.OPENAI_COMPATIBLE_API_KEY) or self._clean(
            self.DEEPSEEK_API_KEY
        )

    @property
    def LLM_BASE_URL(self) -> str:  # noqa: N802
        return self._clean(self.OPENAI_COMPATIBLE_BASE_URL) or self.DEEPSEEK_BASE_URL

    @property
    def LLM_MODEL_NAME(self) -> str:  # noqa: N802
        return self._clean(self.OPENAI_COMPATIBLE_MODEL) or self.DEEPSEEK_MODEL

    @property
    def LLM_PROVIDER_LABEL(self) -> str:  # noqa: N802
        """Human-readable vendor name, shown in the UI and stored on every result."""
        if self._clean(self.OPENAI_COMPATIBLE_API_KEY):
            return "openai-compatible"
        return "deepseek"


@lru_cache
def get_settings() -> Settings:
    """Cached accessor so settings are parsed exactly once per process."""
    return Settings()


settings = get_settings()
