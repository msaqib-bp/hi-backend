"""Prompts and result-shaping shared by every LLM provider.

Two providers ship: Anthropic (Claude) and any OpenAI-compatible endpoint (DeepSeek is
the configured default there). They differ only in *transport* — Claude uses a strict
tool schema, which the API validates for us; OpenAI-compatible endpoints use JSON mode,
which needs the shape described in the prompt and validated on the way back.

Everything that is *not* transport — the triage instructions, the priority rubric, the
label validation, the `AIResult` construction — lives here, so the two providers cannot
quietly drift into giving different answers to the same complaint.
"""

from __future__ import annotations

import json
import re
from abc import abstractmethod
from typing import Any

from app.core.exceptions import AIServiceError
from app.ml.preprocess import extract_keywords
from app.models.enums import AIEngine, ComplaintCategory, ComplaintPriority
from app.services.ai.base import AIAnalyzer, AIResult

CATEGORY_VALUES = [category.value for category in ComplaintCategory]
PRIORITY_VALUES = [priority.value for priority in ComplaintPriority]

TRIAGE_SYSTEM_PROMPT = """You triage municipal complaints for a city corporation's service desk.

Given a citizen's complaint, return:
- category: which service area owns it
- priority: how urgent it is
- summary: ONE line under 25 words telling the field crew what to do and where
- reasoning: one short sentence on why this priority was chosen

Priority guidance:
- critical: immediate danger to life or a major service outage (live wires, missing
  manhole covers, burst mains, sewage in homes, structures about to collapse)
- high: many people affected, or a hazard that will become dangerous soon
- medium: genuine problem, normal repair queue
- low: cosmetic or minor, no safety or service impact

Write the summary as an instruction, not a restatement. Never invent details the
complaint does not contain."""

#: The output contract. Used directly as Claude's tool `input_schema`, and rendered into
#: the prompt for JSON-mode providers — one definition, so the two cannot diverge.
TRIAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORY_VALUES},
        "priority": {"type": "string", "enum": PRIORITY_VALUES},
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
}


def triage_user_content(description: str, location: str | None) -> str:
    content = f"Complaint: {description}"
    if location:
        content += f"\nReported location: {location}"
    return content


def json_mode_instruction() -> str:
    """Extra instruction for providers using JSON mode rather than a tool schema.

    DeepSeek's JSON mode requires the word "json" to appear in the prompt and works far
    more reliably when shown a concrete example alongside the schema, so both are here.
    """
    return (
        "Respond with a single json object and nothing else — no prose, no markdown "
        "fences.\n\n"
        f"Schema:\n{json.dumps(TRIAGE_SCHEMA, indent=2)}\n\n"
        "Example of a valid response:\n"
        '{"category": "drainage", "priority": "critical", '
        '"summary": "Barricade the open manhole on School Road and fit a new cover today.", '
        '"reasoning": "An uncovered manhole on a school route can cause a fall injury."}'
    )


def summary_prompt(
    description: str, category: ComplaintCategory, location: str | None
) -> str:
    return (
        f"Complaint: {description}\n"
        f"Category (already determined): {category.label}\n"
        f"Location: {location or 'not specified'}\n\n"
        "Write ONE line under 25 words telling the field crew what to do and where. "
        "No preamble, no restating the category. Output only that line."
    )


def assistant_prompt(question: str, context: dict[str, Any]) -> str:
    return (
        "You are an analyst for a municipal service desk. Answer the question using "
        "ONLY the live data below. If the data does not contain the answer, say so "
        "plainly rather than guessing. Be concise and specific, cite the numbers, "
        "and keep it under 120 words.\n\n"
        f"LIVE DATA:\n{json.dumps(context, indent=2, default=str)}\n\n"
        f"QUESTION: {question}"
    )


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_payload(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a model response.

    JSON mode usually returns clean JSON, but some providers still wrap it in markdown
    fences or prepend a sentence. Rather than fail the request over formatting, fall back
    to extracting the outermost ``{...}`` block.
    """
    text = (text or "").strip()
    if not text:
        raise AIServiceError("The model returned an empty response.")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK_RE.search(text)
        if not match:
            raise AIServiceError("The model response contained no JSON object.") from None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise AIServiceError(f"The model returned malformed JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise AIServiceError("The model returned JSON that was not an object.")
    return payload


def build_triage_result(
    payload: dict[str, Any],
    *,
    description: str,
    provider: str,
    model_version: str,
    processing_ms: float,
    engine: AIEngine = AIEngine.LLM,
) -> AIResult:
    """Validate a provider's payload and turn it into an ``AIResult``.

    Invalid labels raise rather than being coerced: ``AIPipeline`` catches the error and
    falls back to the ML result, which is strictly better than silently filing a
    complaint under a category the model hallucinated.
    """
    try:
        category = ComplaintCategory(str(payload["category"]).strip().lower())
        priority = ComplaintPriority(str(payload["priority"]).strip().lower())
    except (KeyError, ValueError, AttributeError) as exc:
        raise AIServiceError(f"The model returned an unrecognised label: {exc}") from exc

    notes = [f"Analysed by {provider} ({model_version})."]
    if reasoning := payload.get("reasoning"):
        notes.append(str(reasoning).strip())

    return AIResult(
        category=category,
        priority=priority,
        summary=str(payload.get("summary", "")).strip(),
        engine=engine,
        model_version=model_version,
        provider=provider,
        # Chat-completion APIs expose no calibrated probability for a label, so there is
        # no honest confidence to report. A fixed high value would be a lie; the UI
        # renders 0 as "not available from this engine" and hides the bars.
        category_confidence=0.0,
        priority_confidence=0.0,
        keywords=extract_keywords(description),
        processing_ms=processing_ms,
        notes=notes,
    )


class LLMProvider(AIAnalyzer):
    """Interface the pipeline expects from any language-model engine.

    Beyond ``analyze`` it must offer ``summarize`` (the pipeline's default use — the one
    thing an LLM does better than the extractive fallback) and ``answer_question``
    (the civic assistant).
    """

    #: Short provider identifier stored on every result, e.g. "anthropic" / "deepseek".
    provider: str = "unknown"

    @abstractmethod
    async def summarize(
        self, description: str, category: ComplaintCategory, location: str | None = None
    ) -> str:
        """Return a one-line dispatch summary."""

    @abstractmethod
    async def answer_question(self, question: str, context: dict[str, Any]) -> str:
        """Answer a question grounded on the supplied statistics."""


class NullLLM(LLMProvider):
    """Stand-in used when no provider is configured.

    Reports itself unavailable so the pipeline skips it entirely. Having a real object
    here rather than ``None`` keeps every call site free of null checks.
    """

    name = "none"
    provider = "none"

    @property
    def available(self) -> bool:
        return False

    async def analyze(self, description: str, location: str | None = None) -> AIResult:
        raise AIServiceError("No language-model provider is configured.")

    async def summarize(
        self, description: str, category: ComplaintCategory, location: str | None = None
    ) -> str:
        raise AIServiceError("No language-model provider is configured.")

    async def answer_question(self, question: str, context: dict[str, Any]) -> str:
        raise AIServiceError("No language-model provider is configured.")

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "available": False, "provider": None}
