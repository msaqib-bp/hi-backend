"""``AIPipeline`` — composes the engines and guarantees a result.

The core promise: **analysing a complaint never fails.** A citizen reporting a burst
water main must not see an error page because a model file is missing or an API quota
ran out. The pipeline degrades through three tiers instead:

    MLAnalyzer  ──(optional summary upgrade)──▶  LLMAnalyzer
         │ unavailable / raises                        │ unavailable / raises
         ▼                                             ▼
    RuleAnalyzer  ────────────────────────────────  keep the ML result

Division of labour is deliberate. The ML classifiers keep ownership of category and
priority because they are measurable, calibrated and free. The LLM is used only for the
dispatch summary — the one part it is genuinely better at — so the expensive, failable
dependency sits on the least critical field.
"""

from __future__ import annotations

import time
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.models.enums import AIEngine
from app.services.ai.base import AIAnalyzer, AIResult
from app.services.ai.llm_analyzer import LLMAnalyzer
from app.services.ai.llm_shared import LLMProvider, NullLLM
from app.services.ai.ml_analyzer import MLAnalyzer
from app.services.ai.openai_compat_analyzer import OpenAICompatibleAnalyzer
from app.services.ai.rule_analyzer import RuleAnalyzer

log = get_logger(__name__)


def build_llm_provider() -> LLMProvider:
    """Construct the configured language-model provider.

    Returns a ``NullLLM`` when none is configured, so the pipeline never has to
    null-check and the app runs identically with no key at all.
    """
    provider = settings.active_llm_provider

    if provider == "anthropic":
        return LLMAnalyzer()
    if provider == "openai_compatible":
        return OpenAICompatibleAnalyzer()

    log.info("llm_provider_not_configured", detail="running on the local ML models only")
    return NullLLM()


class AIPipeline(AIAnalyzer):
    """Orchestrates the available engines behind the same ``AIAnalyzer`` interface.

    Being an ``AIAnalyzer`` itself means callers cannot tell whether they hold a single
    engine or a composition — ``ComplaintManager`` just depends on the interface.
    """

    name = "pipeline"

    def __init__(
        self,
        ml_analyzer: MLAnalyzer | None = None,
        llm_analyzer: LLMProvider | None = None,
        rule_analyzer: RuleAnalyzer | None = None,
        *,
        use_llm_for_summary: bool = True,
    ) -> None:
        self.ml = ml_analyzer if ml_analyzer is not None else MLAnalyzer()
        # Which vendor this is (Claude, DeepSeek, none) is decided by configuration in
        # `build_llm_provider`; the pipeline only depends on the LLMProvider interface.
        self.llm = llm_analyzer if llm_analyzer is not None else build_llm_provider()
        self.rules = rule_analyzer if rule_analyzer is not None else RuleAnalyzer()
        self._use_llm_for_summary = use_llm_for_summary

    @property
    def available(self) -> bool:
        # The rule analyzer is always available, so the pipeline always is too.
        return True

    @property
    def llm_provider(self) -> str:
        """The configured vendor name, or "none"."""
        return getattr(self.llm, "provider", "none") if self.llm.available else "none"

    @property
    def active_engine(self) -> str:
        if self.ml.available:
            return f"ml+{self.llm_provider}" if self.llm.available else "ml"
        return self.llm_provider if self.llm.available else "fallback"

    # ------------------------------------------------------------------ analyze
    async def analyze(self, description: str, location: str | None = None) -> AIResult:
        started = time.perf_counter()
        result = await self._classify(description, location)

        if self._use_llm_for_summary and self.llm.available:
            result = await self._upgrade_summary(result, description, location)

        result.processing_ms = (time.perf_counter() - started) * 1000
        log.info(
            "complaint_analysed",
            engine=result.engine.value,
            category=result.category.value,
            priority=result.priority.value,
            confidence=round(result.category_confidence, 3),
            ms=round(result.processing_ms, 1),
        )
        return result

    async def _classify(self, description: str, location: str | None) -> AIResult:
        """Tier 1: the ML engine. Tier 2: the LLM. Tier 3: keyword rules."""
        if self.ml.available:
            try:
                return await self.ml.analyze(description, location)
            except Exception as exc:
                log.warning("ml_analysis_failed", error=str(exc))

        if self.llm.available:
            try:
                result = await self.llm.analyze(description, location)
                result.notes.append(
                    f"The ML models were unavailable; {self.llm_provider} classified this."
                )
                return result
            except Exception as exc:
                log.warning("llm_analysis_failed", provider=self.llm_provider, error=str(exc))

        log.warning("falling_back_to_rules")
        return await self.rules.analyze(description, location)

    async def _upgrade_summary(
        self, result: AIResult, description: str, location: str | None
    ) -> AIResult:
        """Replace the extractive summary with a Claude-written one.

        Failure here is genuinely harmless — the ML summary is already in place, so a
        timeout or a rate limit costs nothing but a slightly less polished sentence.
        """
        provider = self.llm_provider
        try:
            summary = await self.llm.summarize(description, result.category, location)
            if summary:
                result.summary = summary
                result.engine = AIEngine.HYBRID
                result.notes.append(
                    f"Summary written by {provider}; labels from the ML models."
                )
        except Exception as exc:
            log.info("llm_summary_skipped", provider=provider, error=str(exc))
            result.notes.append(
                f"{provider} summary unavailable; using the extractive summary."
            )
        return result

    # ---------------------------------------------------------------- assistant
    async def answer_question(self, question: str, context: dict[str, Any]) -> tuple[str, str]:
        """Answer a natural-language question about the live data.

        Returns ``(answer, engine)``. With no API key this degrades to a deterministic
        summary of the same statistics rather than refusing — the assistant stays useful,
        it just stops being conversational.
        """
        if self.llm.available:
            try:
                return await self.llm.answer_question(question, context), self.llm_provider
            except Exception as exc:
                log.warning("assistant_llm_failed", provider=self.llm_provider, error=str(exc))

        return self._describe_context(question, context), "statistics"

    @staticmethod
    def _describe_context(question: str, context: dict[str, Any]) -> str:
        """Deterministic stand-in for the assistant when no LLM is configured.

        Reads the same statistics the LLM would have received and reports the headline
        figures. Honest about what it is: this is a data summary, not an answer.
        """
        kpis = context.get("kpis", {})
        top_category = context.get("top_category")
        slowest = context.get("slowest_department")

        lines = [
            "The natural-language assistant needs an API key "
            "(ANTHROPIC_API_KEY or DEEPSEEK_API_KEY). "
            "Here is the current data that would have been used to answer "
            f'"{question.strip()}":',
            "",
            f"• {kpis.get('total_complaints', 0)} complaints total, "
            f"{kpis.get('open_complaints', 0)} still open "
            f"({kpis.get('resolution_rate', 0):.0%} resolved).",
        ]
        if top_category:
            lines.append(f"• Most common category: {top_category}.")
        if slowest:
            lines.append(f"• Slowest department: {slowest}.")
        if (median := kpis.get("median_resolution_hours")) is not None:
            lines.append(f"• Median resolution time: {median:.1f} hours.")
        if (overdue := kpis.get("overdue_open", 0)) > 0:
            lines.append(f"• {overdue} open complaints are past their target time.")
        return "\n".join(lines)

    # -------------------------------------------------------------- description
    def describe(self) -> dict[str, Any]:
        ml_info = self.ml.describe()
        return {
            "ml_available": self.ml.available,
            "llm_available": self.llm.available,
            "llm_provider": self.llm_provider,
            "active_engine": self.active_engine,
            "model_version": ml_info.get("model_version") or settings.MODEL_VERSION,
            "trained_at": ml_info.get("trained_at"),
            "training_samples": ml_info.get("training_samples"),
            "macro_f1": ml_info.get("category_macro_f1"),
            "generalization": ml_info.get("generalization"),
            "engines": [self.ml.describe(), self.llm.describe(), self.rules.describe()],
        }


#: Process-wide pipeline. The ML artifacts are large enough that loading them per
#: request would dominate response time, so the instance is shared.
_pipeline: AIPipeline | None = None


def get_ai_pipeline() -> AIPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AIPipeline()
    return _pipeline
