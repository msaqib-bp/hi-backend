"""Tests for the pluggable LLM provider layer.

The app must behave identically whether it is running on Claude, on DeepSeek, or on no
language model at all — only the *quality* of the dispatch summary should differ. These
tests pin that: provider selection, response parsing, and the guarantee that a
misbehaving provider degrades instead of corrupting a complaint.

No network calls. The transport is stubbed; what is under test is our logic.
"""

from __future__ import annotations

import json

import pytest

from app.core.config import Settings
from app.core.exceptions import AIServiceError
from app.models.enums import AIEngine, ComplaintCategory, ComplaintPriority
from app.services.ai.llm_shared import (
    NullLLM,
    build_triage_result,
    json_mode_instruction,
    parse_json_payload,
)
from app.services.ai.openai_compat_analyzer import OpenAICompatibleAnalyzer


class TestProviderSelection:
    """`LLM_PROVIDER=auto` has to do the obvious thing with whatever keys exist."""

    def test_no_keys_means_no_provider(self) -> None:
        settings = Settings(ANTHROPIC_API_KEY=None, DEEPSEEK_API_KEY=None)
        assert settings.active_llm_provider == "none"
        assert settings.llm_enabled is False

    def test_deepseek_key_alone_selects_deepseek(self) -> None:
        settings = Settings(DEEPSEEK_API_KEY="sk-test")
        assert settings.active_llm_provider == "openai_compatible"
        assert settings.llm_enabled is True
        assert settings.LLM_PROVIDER_LABEL == "deepseek"
        assert settings.LLM_BASE_URL == "https://api.deepseek.com"
        assert settings.LLM_MODEL_NAME == "deepseek-chat"

    def test_anthropic_key_alone_selects_anthropic(self) -> None:
        settings = Settings(ANTHROPIC_API_KEY="sk-ant-test")
        assert settings.active_llm_provider == "anthropic"

    def test_both_keys_prefers_anthropic(self) -> None:
        """Claude's strict tool schema cannot return an invalid label, so when both are
        available it is the safer default."""
        settings = Settings(ANTHROPIC_API_KEY="sk-ant-test", DEEPSEEK_API_KEY="sk-test")
        assert settings.active_llm_provider == "anthropic"

    def test_explicit_choice_overrides_auto(self) -> None:
        settings = Settings(
            LLM_PROVIDER="deepseek",
            ANTHROPIC_API_KEY="sk-ant-test",
            DEEPSEEK_API_KEY="sk-test",
        )
        assert settings.active_llm_provider == "openai_compatible"

    def test_explicit_choice_without_its_key_reports_none(self) -> None:
        """A misconfiguration must surface, not silently bill the other vendor."""
        settings = Settings(LLM_PROVIDER="deepseek", ANTHROPIC_API_KEY="sk-ant-test")
        assert settings.active_llm_provider == "none"

    def test_provider_can_be_disabled_outright(self) -> None:
        settings = Settings(LLM_PROVIDER="none", DEEPSEEK_API_KEY="sk-test")
        assert settings.active_llm_provider == "none"

    def test_blank_key_is_not_a_key(self) -> None:
        """Render sets unfilled variables to an empty string rather than unsetting them."""
        settings = Settings(DEEPSEEK_API_KEY="   ")
        assert settings.active_llm_provider == "none"

    def test_generic_endpoint_overrides_deepseek_defaults(self) -> None:
        settings = Settings(
            OPENAI_COMPATIBLE_API_KEY="sk-groq",
            OPENAI_COMPATIBLE_BASE_URL="https://api.groq.com/openai/v1",
            OPENAI_COMPATIBLE_MODEL="llama-3.3-70b",
        )
        assert settings.active_llm_provider == "openai_compatible"
        assert settings.LLM_BASE_URL == "https://api.groq.com/openai/v1"
        assert settings.LLM_MODEL_NAME == "llama-3.3-70b"
        assert settings.LLM_PROVIDER_LABEL == "openai-compatible"


class TestResponseParsing:
    """JSON-mode providers do not always return bare JSON."""

    def test_parses_clean_json(self) -> None:
        assert parse_json_payload('{"category": "water"}') == {"category": "water"}

    def test_recovers_json_from_markdown_fences(self) -> None:
        """A model that wraps its answer in ```json should not cost us the complaint."""
        wrapped = '```json\n{"category": "drainage", "priority": "high"}\n```'
        assert parse_json_payload(wrapped)["category"] == "drainage"

    def test_recovers_json_after_a_preamble_sentence(self) -> None:
        noisy = 'Here is the triage:\n{"category": "waste", "priority": "medium"}'
        assert parse_json_payload(noisy)["category"] == "waste"

    def test_empty_response_raises(self) -> None:
        with pytest.raises(AIServiceError):
            parse_json_payload("")

    def test_non_json_response_raises(self) -> None:
        with pytest.raises(AIServiceError):
            parse_json_payload("I cannot help with that.")

    def test_json_array_is_rejected(self) -> None:
        """A list is valid JSON but not the contract."""
        with pytest.raises(AIServiceError):
            parse_json_payload("[1, 2, 3]")

    def test_json_mode_instruction_contains_the_trigger_word(self) -> None:
        """DeepSeek's JSON mode requires the literal word "json" in the prompt."""
        assert "json" in json_mode_instruction().lower()


class TestTriageResultBuilding:
    def test_valid_payload_builds_a_result(self) -> None:
        result = build_triage_result(
            {
                "category": "drainage",
                "priority": "critical",
                "summary": "Barricade the open manhole on School Road today.",
                "reasoning": "An uncovered manhole on a school route can injure a child.",
            },
            description="The manhole cover is missing near the school gate.",
            provider="deepseek",
            model_version="deepseek-chat",
            processing_ms=812.0,
        )

        assert result.category is ComplaintCategory.DRAINAGE
        assert result.priority is ComplaintPriority.CRITICAL
        assert result.provider == "deepseek"
        assert result.engine is AIEngine.LLM
        assert result.recommended_department == "drainage-sewerage"
        # Chat APIs give no calibrated probability, so none is claimed.
        assert result.category_confidence == 0.0

    def test_labels_are_case_and_whitespace_tolerant(self) -> None:
        result = build_triage_result(
            {"category": " WATER ", "priority": "High", "summary": "Fix the leak."},
            description="Pipeline leaking.",
            provider="deepseek",
            model_version="deepseek-chat",
            processing_ms=1.0,
        )
        assert result.category is ComplaintCategory.WATER
        assert result.priority is ComplaintPriority.HIGH

    def test_hallucinated_category_raises_rather_than_being_coerced(self) -> None:
        """Filing a complaint under an invented category is worse than falling back to
        the ML result, so an unknown label must fail loudly."""
        with pytest.raises(AIServiceError):
            build_triage_result(
                {"category": "potholes_and_stuff", "priority": "high", "summary": "x"},
                description="Road is broken.",
                provider="deepseek",
                model_version="deepseek-chat",
                processing_ms=1.0,
            )

    def test_missing_field_raises(self) -> None:
        with pytest.raises(AIServiceError):
            build_triage_result(
                {"category": "water"},
                description="Leak.",
                provider="deepseek",
                model_version="deepseek-chat",
                processing_ms=1.0,
            )

    def test_result_is_json_serialisable(self) -> None:
        """`ai_output` is persisted as JSON, including the new provider field."""
        result = build_triage_result(
            {"category": "waste", "priority": "low", "summary": "Empty the bin."},
            description="Bin is full.",
            provider="deepseek",
            model_version="deepseek-chat",
            processing_ms=1.0,
        )
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["provider"] == "deepseek"


class TestOpenAICompatibleAnalyzer:
    def test_unconfigured_analyzer_reports_unavailable(self) -> None:
        analyzer = OpenAICompatibleAnalyzer(api_key="")
        assert analyzer.available is False

    def test_configured_analyzer_reports_available(self) -> None:
        analyzer = OpenAICompatibleAnalyzer(api_key="sk-test", model="deepseek-chat")
        assert analyzer.available is True
        assert analyzer.describe()["model"] == "deepseek-chat"

    async def test_unconfigured_analyze_raises(self) -> None:
        with pytest.raises(AIServiceError):
            await OpenAICompatibleAnalyzer(api_key="").analyze("The drain is blocked here.")

    async def test_analyze_parses_a_provider_response(self, monkeypatch) -> None:
        """End-to-end through `analyze`, with only the HTTP call stubbed."""
        analyzer = OpenAICompatibleAnalyzer(api_key="sk-test", provider="deepseek")

        async def fake_chat(**kwargs):
            assert kwargs["json_mode"] is True  # triage must request JSON mode
            return json.dumps(
                {
                    "category": "electricity",
                    "priority": "critical",
                    "summary": "Isolate the sparking transformer near the park immediately.",
                    "reasoning": "A sparking transformer is an immediate electrocution risk.",
                }
            )

        monkeypatch.setattr(analyzer, "_chat", fake_chat)

        result = await analyzer.analyze("The transformer near the park is sparking.")
        assert result.category is ComplaintCategory.ELECTRICITY
        assert result.priority is ComplaintPriority.CRITICAL
        assert result.provider == "deepseek"

    async def test_summary_strips_surrounding_quotes(self, monkeypatch) -> None:
        """Models frequently return a quoted one-liner; the quotes must not reach the UI."""
        analyzer = OpenAICompatibleAnalyzer(api_key="sk-test")

        async def fake_chat(**_kwargs):
            return '"Clear the blocked drain on MG Road before the next rain."'

        monkeypatch.setattr(analyzer, "_chat", fake_chat)
        summary = await analyzer.summarize("Drain blocked.", ComplaintCategory.DRAINAGE)
        assert not summary.startswith('"')
        assert summary.startswith("Clear the blocked drain")


class TestNullProvider:
    """With no key configured the pipeline holds a real object, not None."""

    def test_reports_unavailable(self) -> None:
        assert NullLLM().available is False

    async def test_every_method_raises_rather_than_returning_junk(self) -> None:
        null = NullLLM()
        with pytest.raises(AIServiceError):
            await null.analyze("x")
        with pytest.raises(AIServiceError):
            await null.summarize("x", ComplaintCategory.OTHER)
        with pytest.raises(AIServiceError):
            await null.answer_question("x", {})


class TestPipelineWithProviders:
    async def test_pipeline_falls_back_when_provider_returns_garbage(self) -> None:
        """A provider returning an invalid label must not corrupt the complaint — the ML
        classification stands and only the summary is affected."""
        from app.services.ai.pipeline import AIPipeline
        from tests.conftest import StubAnalyzer

        class BrokenSummaryLLM(NullLLM):
            provider = "deepseek"

            @property
            def available(self) -> bool:
                return True

            async def summarize(self, description, category, location=None):  # noqa: ANN001
                raise AIServiceError("provider exploded")

        pipeline = AIPipeline(
            ml_analyzer=StubAnalyzer(category=ComplaintCategory.WATER),  # type: ignore[arg-type]
            llm_analyzer=BrokenSummaryLLM(),
            use_llm_for_summary=True,
        )

        result = await pipeline.analyze("Water pipeline is leaking on the main road.")
        assert result.category is ComplaintCategory.WATER  # ML result survives
        assert result.summary  # the extractive summary is still there
        assert any("unavailable" in note for note in result.notes)

    async def test_active_engine_names_the_provider(self) -> None:
        from app.services.ai.pipeline import AIPipeline
        from tests.conftest import StubAnalyzer

        class WorkingLLM(NullLLM):
            provider = "deepseek"

            @property
            def available(self) -> bool:
                return True

        pipeline = AIPipeline(
            ml_analyzer=StubAnalyzer(),  # type: ignore[arg-type]
            llm_analyzer=WorkingLLM(),
        )
        assert pipeline.active_engine == "ml+deepseek"
        assert pipeline.llm_provider == "deepseek"
