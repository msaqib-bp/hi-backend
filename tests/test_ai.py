"""AI layer tests.

Two kinds of test live here:

* **Contract and fallback tests** use stubs and always run. They protect the guarantee
  that matters most — analysing a complaint never fails, whatever the engines do.
* **Real-model tests** load the trained artifacts and are skipped when they are absent,
  so a fresh clone that has not run training still gets a green suite.
"""

from __future__ import annotations

import pytest

from app.models.enums import AIEngine, ComplaintCategory, ComplaintPriority
from app.services.ai.pipeline import AIPipeline
from app.services.ai.rule_analyzer import RuleAnalyzer
from tests.conftest import StubAnalyzer


class TestAIResultContract:
    async def test_result_routes_to_a_department(self) -> None:
        """Every result must name an owning department — routing is not optional."""
        result = await StubAnalyzer(category=ComplaintCategory.DRAINAGE).analyze("x" * 30)
        assert result.recommended_department == "drainage-sewerage"

    async def test_result_serialises_to_json_safe_primitives(self) -> None:
        """``ai_output`` is persisted as JSON, so no enums or numpy scalars may leak."""
        import json

        result = await StubAnalyzer().analyze("Water leaking on the main road here.")
        payload = result.to_dict()

        json.dumps(payload)  # raises if anything is not JSON-serialisable
        assert isinstance(payload["category"], str)
        assert isinstance(payload["priority"], str)
        assert isinstance(payload["engine"], str)

    async def test_low_confidence_is_flagged_for_review(self) -> None:
        result = await StubAnalyzer().analyze("x" * 30)
        result.category_confidence = 0.3
        assert result.is_confident is False


class TestPipelineFallback:
    """The core promise: a citizen's complaint is never lost to an AI failure."""

    async def test_uses_ml_when_available(self) -> None:
        ml = StubAnalyzer(category=ComplaintCategory.WASTE)
        pipeline = AIPipeline(
            ml_analyzer=ml,  # type: ignore[arg-type]
            llm_analyzer=StubAnalyzer(available=False),  # type: ignore[arg-type]
            use_llm_for_summary=False,
        )
        result = await pipeline.analyze("Garbage is piling up near the market.")
        assert result.category is ComplaintCategory.WASTE
        assert ml.calls == 1

    async def test_falls_back_to_llm_when_ml_unavailable(self) -> None:
        llm = StubAnalyzer(category=ComplaintCategory.SAFETY)
        pipeline = AIPipeline(
            ml_analyzer=StubAnalyzer(available=False),  # type: ignore[arg-type]
            llm_analyzer=llm,  # type: ignore[arg-type]
            use_llm_for_summary=False,
        )
        result = await pipeline.analyze("The wall is about to collapse near the school.")
        assert result.category is ComplaintCategory.SAFETY
        assert llm.calls == 1

    async def test_falls_back_to_rules_when_both_engines_fail(self) -> None:
        """Both engines down must still produce a usable, honestly-labelled result."""
        pipeline = AIPipeline(
            ml_analyzer=StubAnalyzer(raises=True),  # type: ignore[arg-type]
            llm_analyzer=StubAnalyzer(raises=True),  # type: ignore[arg-type]
            use_llm_for_summary=False,
        )
        result = await pipeline.analyze(
            "The manhole cover is missing on the main road and it is very dangerous."
        )
        assert result.engine is AIEngine.FALLBACK
        assert result.category is ComplaintCategory.DRAINAGE  # lexicon still routes it
        assert any("review" in note.lower() for note in result.notes)

    async def test_ml_exception_does_not_propagate(self) -> None:
        """An analyzer raising must never surface as a failed submission."""
        pipeline = AIPipeline(
            ml_analyzer=StubAnalyzer(raises=True),  # type: ignore[arg-type]
            llm_analyzer=StubAnalyzer(available=False),  # type: ignore[arg-type]
            use_llm_for_summary=False,
        )
        result = await pipeline.analyze("Streetlight is not working on our lane at all.")
        assert result is not None

    async def test_pipeline_is_always_available(self) -> None:
        pipeline = AIPipeline(
            ml_analyzer=StubAnalyzer(available=False),  # type: ignore[arg-type]
            llm_analyzer=StubAnalyzer(available=False),  # type: ignore[arg-type]
        )
        assert pipeline.available is True
        assert pipeline.active_engine == "fallback"

    async def test_assistant_degrades_to_statistics_digest(self) -> None:
        """With no LLM the assistant reports real numbers instead of refusing."""
        pipeline = AIPipeline(
            ml_analyzer=StubAnalyzer(),  # type: ignore[arg-type]
            llm_analyzer=StubAnalyzer(available=False),  # type: ignore[arg-type]
        )
        context = {
            "kpis": {
                "total_complaints": 42,
                "open_complaints": 7,
                "resolution_rate": 0.83,
                "median_resolution_hours": 18.5,
                "overdue_open": 3,
            },
            "top_category": "Water Supply",
            "slowest_department": "Public Works",
        }
        answer, engine = await pipeline.answer_question("Which team is slowest?", context)
        assert engine == "statistics"
        assert "42" in answer
        assert "Public Works" in answer


class TestRuleAnalyzer:
    """The safety net must be genuinely useful, not just non-crashing."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("The drain is blocked and sewage is overflowing", ComplaintCategory.DRAINAGE),
            ("Garbage bin overflowing with rubbish", ComplaintCategory.WASTE),
            ("Streetlight and electricity pole not working", ComplaintCategory.ELECTRICITY),
            ("Large pothole on the road surface", ComplaintCategory.ROAD),
            ("No drinking water supply from the tap", ComplaintCategory.WATER),
        ],
    )
    async def test_lexicon_routes_common_complaints(
        self, text: str, expected: ComplaintCategory
    ) -> None:
        result = await RuleAnalyzer().analyze(text)
        assert result.category is expected

    async def test_confidence_is_capped_so_weak_guesses_get_reviewed(self) -> None:
        result = await RuleAnalyzer().analyze("The drain is blocked near the market.")
        assert result.category_confidence <= 0.5

    async def test_unmatched_text_falls_to_other(self) -> None:
        result = await RuleAnalyzer().analyze("Something is wrong somewhere please help.")
        assert result.category is ComplaintCategory.OTHER

    async def test_critical_markers_escalate_priority(self) -> None:
        result = await RuleAnalyzer().analyze(
            "A live wire is sparking and there was a fire near the pole."
        )
        assert result.priority is ComplaintPriority.CRITICAL


# --------------------------------------------------------------- real-model tests
def _artifacts_present() -> bool:
    from app.core.config import settings
    from app.ml.constants import CATEGORY_MODEL_FILE

    return (settings.ML_ARTIFACT_DIR / CATEGORY_MODEL_FILE).exists()


requires_model = pytest.mark.skipif(
    not _artifacts_present(),
    reason="Trained artifacts missing — run `python -m app.ml.train`.",
)


@requires_model
class TestTrainedModels:
    """Behavioural checks against the real classifiers.

    These assert *routing outcomes a municipal officer would agree with*, not exact
    probabilities — probabilities shift on every retrain, and a test that pins them
    would fail for no useful reason.
    """

    @pytest.fixture(scope="class")
    def analyzer(self):
        from app.services.ai.ml_analyzer import MLAnalyzer

        return MLAnalyzer()

    async def test_models_load(self, analyzer) -> None:
        assert analyzer.available is True

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (
                "There is a large water leak from the pipeline and no water in our taps.",
                ComplaintCategory.WATER,
            ),
            (
                "The manhole cover is missing and sewage is flowing onto the street.",
                ComplaintCategory.DRAINAGE,
            ),
            (
                "Garbage has not been collected for four days and the bin is overflowing.",
                ComplaintCategory.WASTE,
            ),
            (
                "The streetlight near our house has been off and the transformer is sparking.",
                ComplaintCategory.ELECTRICITY,
            ),
            (
                "There is a deep pothole in the road surface and vehicles are getting damaged.",
                ComplaintCategory.ROAD,
            ),
        ],
    )
    async def test_classifies_unambiguous_complaints(
        self, analyzer, text: str, expected: ComplaintCategory
    ) -> None:
        result = await analyzer.analyze(text)
        assert result.category is expected

    async def test_dangerous_complaint_is_escalated(self, analyzer) -> None:
        """A live wire over a school route must not be filed as routine."""
        result = await analyzer.analyze(
            "A live electric wire is hanging low over the school gate and children "
            "pass under it every day. This is extremely dangerous."
        )
        assert result.priority in (ComplaintPriority.HIGH, ComplaintPriority.CRITICAL)

    async def test_trivial_complaint_is_not_escalated(self, analyzer) -> None:
        result = await analyzer.analyze(
            "The road markings near the corner have faded slightly. Not urgent at all, "
            "just adding it to your list for future maintenance whenever convenient."
        )
        assert result.priority in (ComplaintPriority.LOW, ComplaintPriority.MEDIUM)

    async def test_summary_is_not_a_negation_fragment(self, analyzer) -> None:
        """Regression: the extractive summariser used to pick 'Not urgent.' as the key
        sentence, because 'urgent' is an urgency term, discarding the actual problem."""
        result = await analyzer.analyze(
            "The streetlight bulb outside my house has fused. Not urgent."
        )
        assert "streetlight" in result.summary.lower()

    async def test_confidence_is_a_probability(self, analyzer) -> None:
        result = await analyzer.analyze("The drain near the market is completely blocked.")
        assert 0.0 <= result.category_confidence <= 1.0
        assert 0.0 <= result.priority_confidence <= 1.0

    async def test_alternatives_rank_below_the_prediction(self, analyzer) -> None:
        result = await analyzer.analyze("Sewage water is overflowing from the manhole.")
        for alternative in result.category_alternatives:
            assert alternative.confidence <= result.category_confidence

    async def test_handles_messy_real_world_input(self, analyzer) -> None:
        """Lowercase, no punctuation, typos — how complaints actually arrive."""
        result = await analyzer.analyze(
            "garbge is not collected frm our lane since 4 days its stinkng badly"
        )
        assert result.category is ComplaintCategory.WASTE

    async def test_empty_ish_input_does_not_crash(self, analyzer) -> None:
        result = await analyzer.analyze("problem")
        assert result.category in list(ComplaintCategory)


class TestSummaryFormatting:
    """Regression tests for summaries that reach a citizen's screen.

    Complaints arrive pasted from chat apps and email, so trailing quotes and stray
    punctuation are normal input, not edge cases.
    """

    @pytest.fixture(scope="class")
    def analyzer(self):
        from app.services.ai.ml_analyzer import MLAnalyzer

        return MLAnalyzer()

    @requires_model
    async def test_trailing_quote_does_not_survive_into_the_summary(self, analyzer) -> None:
        """Reported from a real submission: a description ending in `."` produced
        `...becoming difficult.".` — a stray quote and a doubled period."""
        result = await analyzer.analyze(
            'There is a large water leak near the main road and traffic is becoming difficult."',
            location="hyderabad",
        )
        assert '."' not in result.summary
        assert ".." not in result.summary
        assert result.summary.endswith(".")

    @requires_model
    async def test_fully_quoted_complaint_is_unwrapped(self, analyzer) -> None:
        result = await analyzer.analyze('"The drain near the market is completely blocked."')
        assert not result.summary.split(": ", 1)[-1].startswith('"')

    @requires_model
    async def test_blank_location_produces_no_dangling_preposition(self, analyzer) -> None:
        result = await analyzer.analyze("The streetlight has been off for two weeks.", "   ")
        assert " at :" not in result.summary
        assert " at ." not in result.summary


class TestSentenceTidier:
    """Unit-level checks for the shared cleaner."""

    def test_strips_trailing_punctuation_and_quotes(self) -> None:
        from app.ml.preprocess import tidy_sentence

        assert tidy_sentence('Water is leaking badly."') == "Water is leaking badly"
        assert tidy_sentence("Water is leaking badly...") == "Water is leaking badly"
        assert tidy_sentence("Water is leaking badly!?") == "Water is leaking badly"

    def test_unwraps_a_fully_quoted_sentence(self) -> None:
        from app.ml.preprocess import tidy_sentence

        assert tidy_sentence('"Water is leaking badly."') == "Water is leaking badly"

    def test_truncates_on_a_word_boundary(self) -> None:
        from app.ml.preprocess import tidy_sentence

        long_text = "word " * 60
        out = tidy_sentence(long_text, max_length=40)
        assert len(out) <= 40
        assert out.endswith("…")

    def test_handles_empty_input(self) -> None:
        from app.ml.preprocess import tidy_sentence

        assert tidy_sentence("") == ""
        assert tidy_sentence("   ") == ""
