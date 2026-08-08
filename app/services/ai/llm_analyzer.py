"""The Claude analyzer — the optional upgrade engine.

Enabled only when ``ANTHROPIC_API_KEY`` is set. It produces noticeably better summaries
than the extractive fallback (it can compress three rambling sentences into one crisp
dispatch instruction) and it powers the civic assistant.

It is deliberately *not* the default. The Claude API is pay-per-token with no free tier,
so making it mandatory would mean the demo dies when a key expires or a quota runs out.
``AIPipeline`` treats it as an enhancement layer over a result the ML engine already
produced.
"""

from __future__ import annotations

import json
import time
from typing import Any

from app.core.config import settings
from app.core.exceptions import AIServiceError
from app.core.logging import get_logger
from app.ml.preprocess import extract_keywords
from app.models.enums import AIEngine, ComplaintCategory, ComplaintPriority
from app.services.ai.base import AIAnalyzer, AIResult

log = get_logger(__name__)

_CATEGORY_VALUES = [category.value for category in ComplaintCategory]
_PRIORITY_VALUES = [priority.value for priority in ComplaintPriority]

SYSTEM_PROMPT = """You triage municipal complaints for a city corporation's service desk.

Given a citizen's complaint, return:
- category: which service area owns it
- priority: how urgent it is
- summary: ONE line under 25 words telling the field crew what to do and where

Priority guidance:
- critical: immediate danger to life or a major service outage (live wires, missing
  manhole covers, burst mains, sewage in homes, structures about to collapse)
- high: many people affected, or a hazard that will become dangerous soon
- medium: genuine problem, normal repair queue
- low: cosmetic or minor, no safety or service impact

Write the summary as an instruction, not a restatement. Never invent details the
complaint does not contain."""


def _build_tool_schema() -> dict[str, Any]:
    """Strict tool schema — guarantees parseable, valid-enum output.

    Using a tool with ``strict: true`` rather than asking for JSON in prose means the
    API validates the shape for us; there is no partial-JSON parsing to get wrong.
    """
    return {
        "name": "record_triage",
        "description": "Record the triage decision for a civic complaint.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": _CATEGORY_VALUES},
                "priority": {"type": "string", "enum": _PRIORITY_VALUES},
                "summary": {
                    "type": "string",
                    "description": "One actionable line for the field crew, under 25 words.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "One short sentence on why this priority was chosen.",
                },
            },
            "required": ["category", "priority", "summary", "reasoning"],
            "additionalProperties": False,
        },
    }


class LLMAnalyzer(AIAnalyzer):
    """Claude-backed analyzer. Optional; never required for the app to function."""

    name = "llm"

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self._api_key = api_key or settings.ANTHROPIC_API_KEY
        self._model = model or settings.LLM_MODEL
        self._client: Any = None

        if self._api_key:
            try:
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic(
                    api_key=self._api_key, timeout=settings.LLM_TIMEOUT_SECONDS, max_retries=1
                )
                log.info("llm_analyzer_ready", model=self._model)
            except Exception as exc:  # pragma: no cover - SDK import/config failure
                log.warning("llm_analyzer_init_failed", error=str(exc))
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    # ---------------------------------------------------------------- inference
    async def analyze(self, description: str, location: str | None = None) -> AIResult:
        if not self.available:
            raise AIServiceError("The LLM engine is not configured (no ANTHROPIC_API_KEY).")

        started = time.perf_counter()
        user_content = f"Complaint: {description}"
        if location:
            user_content += f"\nReported location: {location}"

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=settings.LLM_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=[_build_tool_schema()],
                tool_choice={"type": "tool", "name": "record_triage"},
                messages=[{"role": "user", "content": user_content}],
            )
        except Exception as exc:
            log.warning("llm_request_failed", error=str(exc))
            raise AIServiceError(f"The LLM request failed: {exc}") from exc

        payload = self._extract_tool_input(response)
        elapsed_ms = (time.perf_counter() - started) * 1000

        try:
            category = ComplaintCategory(payload["category"])
            priority = ComplaintPriority(payload["priority"])
        except (KeyError, ValueError) as exc:
            raise AIServiceError(f"The LLM returned an unrecognised label: {exc}") from exc

        notes = ["Analysed by Claude."]
        if reasoning := payload.get("reasoning"):
            notes.append(str(reasoning))

        return AIResult(
            category=category,
            priority=priority,
            summary=str(payload.get("summary", "")).strip(),
            engine=AIEngine.LLM,
            model_version=self._model,
            # The API does not expose token-level probabilities, so there is no honest
            # confidence to report. A fixed high value would be a lie; these are the
            # values the UI renders as "not available from this engine".
            category_confidence=0.0,
            priority_confidence=0.0,
            category_alternatives=[],
            priority_alternatives=[],
            keywords=extract_keywords(description),
            processing_ms=elapsed_ms,
            notes=notes,
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

        raise AIServiceError("The LLM response contained no triage tool call.")

    # -------------------------------------------------------------- enhancement
    async def summarize(
        self, description: str, category: ComplaintCategory, location: str | None = None
    ) -> str:
        """Generate just the dispatch summary, leaving the ML labels in place.

        This is the pipeline's default use of the LLM: the classifiers are measurable
        and calibrated, so they keep ownership of category and priority; the LLM only
        does the thing it is genuinely better at, which is writing the one-line
        instruction a crew reads.
        """
        if not self.available:
            raise AIServiceError("The LLM engine is not configured.")

        prompt = (
            f"Complaint: {description}\n"
            f"Category (already determined): {category.label}\n"
            f"Location: {location or 'not specified'}\n\n"
            "Write ONE line under 25 words telling the field crew what to do and where. "
            "No preamble, no restating the category. Output only that line."
        )
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise AIServiceError(f"The LLM summary request failed: {exc}") from exc

        if getattr(response, "stop_reason", None) == "refusal":
            raise AIServiceError("The model declined to summarise this complaint.")

        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                return block.text.strip()
        raise AIServiceError("The LLM returned an empty summary.")

    async def answer_question(self, question: str, context: dict[str, Any]) -> str:
        """Back the civic assistant, grounded on live statistics.

        ``context`` is the real aggregate data from ``StatisticsService``. Passing it in
        means the model summarises numbers it was given rather than inventing them —
        and the same context is returned to the caller so the answer stays auditable.
        """
        if not self.available:
            raise AIServiceError("The LLM engine is not configured.")

        prompt = (
            "You are an analyst for a municipal service desk. Answer the question using "
            "ONLY the live data below. If the data does not contain the answer, say so "
            "plainly rather than guessing. Be concise and specific, cite the numbers, "
            "and keep it under 120 words.\n\n"
            f"LIVE DATA:\n{json.dumps(context, indent=2, default=str)}\n\n"
            f"QUESTION: {question}"
        )
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=settings.LLM_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise AIServiceError(f"The assistant request failed: {exc}") from exc

        if getattr(response, "stop_reason", None) == "refusal":
            raise AIServiceError("The model declined to answer this question.")

        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                return block.text.strip()
        raise AIServiceError("The assistant returned an empty answer.")

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "available": self.available, "model": self._model}
