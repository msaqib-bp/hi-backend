"""Analyzer for OpenAI-compatible chat-completion endpoints.

Configured for **DeepSeek** by default, but nothing here is DeepSeek-specific: the base
URL, model and key are all settings, so the same class serves Together, Groq, OpenRouter,
a local vLLM/Ollama server, or OpenAI itself. Point `LLM_BASE_URL` somewhere else and it
works.

**Why raw HTTP rather than an SDK.** The surface used here is one endpoint —
``POST /chat/completions`` — and ``httpx`` is already a dependency. Adding the `openai`
package to reach a non-OpenAI vendor would buy nothing but another thing to pin.

**Why JSON mode rather than function calling.** Tool-calling support across
OpenAI-compatible vendors is uneven, and a provider that silently ignores the tool
returns prose that fails to parse. JSON mode plus an explicit schema and example in the
prompt (see ``llm_shared.json_mode_instruction``) is the most portable contract, and the
response is validated on the way back regardless.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from app.core.config import settings
from app.core.exceptions import AIServiceError
from app.core.logging import get_logger
from app.models.enums import ComplaintCategory
from app.services.ai.base import AIResult
from app.services.ai.llm_shared import (
    TRIAGE_SYSTEM_PROMPT,
    LLMProvider,
    assistant_prompt,
    build_triage_result,
    json_mode_instruction,
    parse_json_payload,
    summary_prompt,
    triage_user_content,
)

log = get_logger(__name__)


class OpenAICompatibleAnalyzer(LLMProvider):
    """Chat-completion analyzer for any OpenAI-compatible API."""

    name = "llm"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        self._api_key = (api_key or settings.LLM_API_KEY or "").strip()
        self._base_url = (base_url or settings.LLM_BASE_URL).rstrip("/")
        self._model = model or settings.LLM_MODEL_NAME
        self.provider = provider or settings.LLM_PROVIDER_LABEL

        if self._api_key:
            log.info(
                "openai_compatible_analyzer_ready",
                provider=self.provider,
                model=self._model,
                base_url=self._base_url,
            )

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    # ------------------------------------------------------------------ transport
    async def _chat(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> str:
        """One chat-completion round trip. Returns the assistant's text content."""
        if not self.available:
            raise AIServiceError(f"The {self.provider} engine is not configured (no API key).")

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise AIServiceError(f"The {self.provider} request timed out.") from exc
        except httpx.HTTPError as exc:
            raise AIServiceError(f"Could not reach {self.provider}: {exc}") from exc

        if response.status_code != 200:
            # Surface the vendor's own message — "insufficient balance" and "invalid key"
            # are both common and need different fixes, so collapsing them to
            # "request failed" would waste the operator's time.
            detail = response.text[:300]
            raise AIServiceError(
                f"{self.provider} returned HTTP {response.status_code}: {detail}"
            )

        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError, TypeError) as exc:
            raise AIServiceError(
                f"Unexpected response shape from {self.provider}: {exc}"
            ) from exc

        if not content or not str(content).strip():
            raise AIServiceError(f"{self.provider} returned an empty response.")
        return str(content)

    # ------------------------------------------------------------------ inference
    async def analyze(self, description: str, location: str | None = None) -> AIResult:
        started = time.perf_counter()

        content = await self._chat(
            messages=[
                {
                    "role": "system",
                    "content": f"{TRIAGE_SYSTEM_PROMPT}\n\n{json_mode_instruction()}",
                },
                {"role": "user", "content": triage_user_content(description, location)},
            ],
            max_tokens=settings.LLM_MAX_TOKENS,
            # Low temperature: triage is a classification decision, and run-to-run
            # variation on the same complaint would be a defect, not creativity.
            temperature=0.2,
            json_mode=True,
        )

        return build_triage_result(
            parse_json_payload(content),
            description=description,
            provider=self.provider,
            model_version=self._model,
            processing_ms=(time.perf_counter() - started) * 1000,
        )

    async def summarize(
        self, description: str, category: ComplaintCategory, location: str | None = None
    ) -> str:
        content = await self._chat(
            messages=[
                {"role": "user", "content": summary_prompt(description, category, location)}
            ],
            max_tokens=200,
            temperature=0.3,
        )
        # Models sometimes wrap a one-liner in quotes; strip them so the summary reads
        # naturally in the dispatch card.
        return content.strip().strip('"').strip()

    async def answer_question(self, question: str, context: dict[str, Any]) -> str:
        content = await self._chat(
            messages=[{"role": "user", "content": assistant_prompt(question, context)}],
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=0.4,
        )
        return content.strip()

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "provider": self.provider,
            "model": self._model,
            "base_url": self._base_url,
        }
