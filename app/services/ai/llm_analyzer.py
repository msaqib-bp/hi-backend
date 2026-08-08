"""The Claude analyzer — one of two optional LLM providers.

Enabled when ``ANTHROPIC_API_KEY`` is set. It produces noticeably better summaries than
the extractive fallback (it can compress three rambling sentences into one crisp dispatch
instruction) and it powers the civic assistant.

It is deliberately *not* the default engine. The Claude API is pay-per-token with no free
tier, so making it mandatory would mean the demo dies when a key expires or a quota runs
out. ``AIPipeline`` treats it as an enhancement layer over a result the ML engine already
produced.

Unlike the OpenAI-compatible provider, this one uses a **strict tool schema** rather than
JSON mode: the API validates the shape and the enum values server-side, so there is no
partial-JSON parsing to get wrong. Prompts and result construction are shared between
both providers via ``llm_shared``.
"""

from __future__ import annotations

import json
import time
from typing import Any

from app.core.config import settings
from app.core.exceptions import AIServiceError
from app.core.logging import get_logger
from app.models.enums import ComplaintCategory
from app.services.ai.base import AIResult
from app.services.ai.llm_shared import (
    TRIAGE_SCHEMA,
    TRIAGE_SYSTEM_PROMPT,
    LLMProvider,
    assistant_prompt,
    build_triage_result,
    summary_prompt,
    triage_user_content,
)

log = get_logger(__name__)

#: Strict tool definition. `strict: True` makes the API enforce the schema — including
#: the category/priority enums — so an invalid label cannot reach us.
TRIAGE_TOOL: dict[str, Any] = {
    "name": "record_triage",
    "description": "Record the triage decision for a civic complaint.",
    "strict": True,
    "input_schema": TRIAGE_SCHEMA,
}


class LLMAnalyzer(LLMProvider):
    """Claude-backed analyzer. Optional; never required for the app to function."""

    name = "llm"
    provider = "anthropic"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key or settings.ANTHROPIC_API_KEY
        self._model = model or settings.ANTHROPIC_MODEL
        self._client: Any = None

        if self._api_key:
            try:
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic(
                    api_key=self._api_key, timeout=settings.LLM_TIMEOUT_SECONDS, max_retries=1
                )
                log.info("llm_analyzer_ready", provider=self.provider, model=self._model)
            except Exception as exc:  # pragma: no cover - SDK import/config failure
                log.warning("llm_analyzer_init_failed", error=str(exc))
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    # ---------------------------------------------------------------- inference
    async def analyze(self, description: str, location: str | None = None) -> AIResult:
        if not self.available:
            raise AIServiceError("The Claude engine is not configured (no ANTHROPIC_API_KEY).")

        started = time.perf_counter()
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=settings.LLM_MAX_TOKENS,
                system=TRIAGE_SYSTEM_PROMPT,
                tools=[TRIAGE_TOOL],
                tool_choice={"type": "tool", "name": "record_triage"},
                messages=[
                    {"role": "user", "content": triage_user_content(description, location)}
                ],
            )
        except Exception as exc:
            log.warning("llm_request_failed", error=str(exc))
            raise AIServiceError(f"The Claude request failed: {exc}") from exc

        return build_triage_result(
            self._extract_tool_input(response),
            description=description,
            provider=self.provider,
            model_version=self._model,
            processing_ms=(time.perf_counter() - started) * 1000,
        )

    @staticmethod
    def _extract_tool_input(response: Any) -> dict[str, Any]:
        """Pull the tool-call arguments out of the response.

        ``stop_reason == "refusal"`` must be checked before reading content — on a
        refusal the content list is empty and indexing it would raise.
        """
        if getattr(response, "stop_reason", None) == "refusal":
            raise AIServiceError("The model declined to analyse this complaint.")

        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "tool_use":
                block_input = getattr(block, "input", None)
                if isinstance(block_input, dict):
                    return block_input
                if isinstance(block_input, str):
                    return json.loads(block_input)

        raise AIServiceError("The Claude response contained no triage tool call.")

    @staticmethod
    def _extract_text(response: Any, what: str) -> str:
        if getattr(response, "stop_reason", None) == "refusal":
            raise AIServiceError(f"The model declined to {what}.")
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                return block.text.strip()
        raise AIServiceError(f"Claude returned an empty {what} response.")

    # -------------------------------------------------------------- enhancement
    async def summarize(
        self, description: str, category: ComplaintCategory, location: str | None = None
    ) -> str:
        """Generate just the dispatch summary, leaving the ML labels in place.

        This is the pipeline's default use of an LLM: the classifiers are measurable and
        calibrated, so they keep ownership of category and priority; the LLM only does the
        thing it is genuinely better at, which is writing the one-line crew instruction.
        """
        if not self.available:
            raise AIServiceError("The Claude engine is not configured.")

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=200,
                messages=[
                    {
                        "role": "user",
                        "content": summary_prompt(description, category, location),
                    }
                ],
            )
        except Exception as exc:
            raise AIServiceError(f"The Claude summary request failed: {exc}") from exc

        return self._extract_text(response, "summarise this complaint")

    async def answer_question(self, question: str, context: dict[str, Any]) -> str:
        """Back the civic assistant, grounded on live statistics.

        ``context`` is the real aggregate data from ``StatisticsService``. Passing it in
        means the model summarises numbers it was given rather than inventing them — and
        the same context is returned to the caller so the answer stays auditable.
        """
        if not self.available:
            raise AIServiceError("The Claude engine is not configured.")

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=settings.LLM_MAX_TOKENS,
                messages=[{"role": "user", "content": assistant_prompt(question, context)}],
            )
        except Exception as exc:
            raise AIServiceError(f"The assistant request failed: {exc}") from exc

        return self._extract_text(response, "answer this question")

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "provider": self.provider,
            "model": self._model,
        }
