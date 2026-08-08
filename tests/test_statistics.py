"""Statistics tests.

Every expected value here is computed by hand in the docstring or comment, so a failure
means the code is wrong rather than that a fixture drifted. These are the numbers the
Batch 4 benchmark is graded on, so they need to be right, not merely self-consistent.
"""

from __future__ import annotations

import pytest

from app.services.statistics import MIN_SAMPLES_FOR_QUARTILES, StatisticsService

#: Textbook sample with a deliberate outlier.
#: sorted: [2, 4, 4, 4, 5, 5, 7, 9]  n = 8  sum = 40
#:   mean   = 40 / 8 = 5.0
#:   median = (4 + 5) / 2 = 4.5
#:   mode   = 4 (appears three times)
#:   range  = 9 - 2 = 7
#:   sample variance = Σ(x-x̄)² / (n-1) = 32 / 7 = 4.571…
#:   sample std dev  = √4.571… = 2.138…
SAMPLE = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]


class TestDescriptiveStats:
    def test_central_tendency_and_dispersion(self) -> None:
        result = StatisticsService.describe(SAMPLE)

        assert result.count == 8
        assert result.mean == 5.0
        assert result.median == 4.5
        assert result.mode == 4.0
        assert result.minimum == 2.0
        assert result.maximum == 9.0
        assert result.range == 7.0
        assert result.variance == pytest.approx(4.57, abs=0.01)
        assert result.std_deviation == pytest.approx(2.14, abs=0.01)

    def test_interpretation_is_generated_and_mentions_the_numbers(self) -> None:
        """The benchmark requires explanation, not just values."""
        result = StatisticsService.describe(SAMPLE)
        assert result.interpretation
        assert "5.0" in result.interpretation  # the mean
        assert "4.5" in result.interpretation  # the median

    def test_empty_sample_does_not_fabricate_zeros(self) -> None:
        """An empty dataset must say so rather than report a confident 0.0."""
        result = StatisticsService.describe([])
        assert result.count == 0
        assert result.mean is None
        assert result.variance is None
        assert "no resolved complaints" in result.interpretation.lower()

    def test_single_value_reports_no_dispersion(self) -> None:
        """Variance of one observation is undefined — it must not be reported as 0."""
        result = StatisticsService.describe([12.0])
        assert result.count == 1
        assert result.mean == 12.0
        assert result.variance is None
        assert result.std_deviation is None

    def test_right_skew_is_detected_and_explained(self) -> None:
        """Mean well above median means a slow tail — the interpretation must say so."""
        skewed = [1.0, 1.0, 2.0, 2.0, 3.0, 200.0]  # mean 34.83, median 2.0
        result = StatisticsService.describe(skewed)
        assert result.mean > result.median
        assert "dragging the average up" in result.interpretation

    def test_symmetric_distribution_is_described_as_such(self) -> None:
        result = StatisticsService.describe([10.0, 10.0, 10.0, 10.0])
        assert "symmetric" in result.interpretation


class TestQuartiles:
    def test_quartiles_and_tukey_fences(self) -> None:
        """numpy linear interpolation over [2,4,4,4,5,5,7,9], n=8:
        Q1 index = 0.25·7 = 1.75 -> between 4 and 4      -> 4.0
        Q2 index = 0.50·7 = 3.5  -> between 4 and 5      -> 4.5
        Q3 index = 0.75·7 = 5.25 -> 5 + 0.25·(7-5)       -> 5.5
        IQR = 5.5 - 4.0 = 1.5
        fences = 4.0 - 1.5·1.5 = 1.75  and  5.5 + 1.5·1.5 = 7.75
        Only 9 exceeds the upper fence, so exactly one outlier.
        """
        result = StatisticsService.quartiles(SAMPLE)

        assert result.q1 == 4.0
        assert result.q2_median == 4.5
        assert result.q3 == 5.5
        assert result.iqr == 1.5
        assert result.lower_fence == 1.75
        assert result.upper_fence == 7.75
        assert result.outlier_count == 1

    def test_outlier_references_are_returned(self) -> None:
        """The dashboard names the slow complaints, so the reference must come back."""
        references = [f"CIV-{index:06d}" for index in range(len(SAMPLE))]
        result = StatisticsService.quartiles(SAMPLE, references)
        # 9.0 is the last element, so its reference is the last one.
        assert result.outlier_references == ["CIV-000007"]

    def test_too_few_samples_refuses_to_compute(self) -> None:
        """Quartiles over n<4 are an artefact of interpolation, not a property of
        the data — the service must decline rather than return a quotable number."""
        result = StatisticsService.quartiles([5.0, 10.0, 15.0])
        assert result.q1 is None
        assert result.iqr is None
        assert str(MIN_SAMPLES_FOR_QUARTILES) in result.interpretation

    def test_no_outliers_is_stated_explicitly(self) -> None:
        result = StatisticsService.quartiles([10.0, 11.0, 12.0, 13.0, 14.0])
        assert result.outlier_count == 0
        assert "no complaint exceeds" in result.interpretation.lower()


class TestDatabaseBackedStatistics:
    async def test_overview_on_empty_database(self, seeded_session) -> None:
        service = StatisticsService(seeded_session)
        overview = await service.overview()

        assert overview.kpis.total_complaints == 0
        assert overview.kpis.resolution_rate == 0.0
        assert "no complaints" in overview.headline.lower()

    async def test_overview_counts_and_rate(self, seeded_session) -> None:
        from app.schemas.complaint import ComplaintCreate, ComplaintUpdate
        from app.services.ai.pipeline import AIPipeline
        from app.services.complaint_manager import ComplaintManager
        from tests.conftest import StubAnalyzer

        pipeline = AIPipeline(
            ml_analyzer=StubAnalyzer(),  # type: ignore[arg-type]
            llm_analyzer=StubAnalyzer(available=False),  # type: ignore[arg-type]
            use_llm_for_summary=False,
        )
        manager = ComplaintManager(seeded_session, pipeline)

        created = []
        for index in range(4):
            complaint, _ = await manager.create(
                ComplaintCreate(
                    description=f"Water pipeline leaking badly at site number {index} here.",
                    location="Test Road",
                ),
                check_duplicates=False,
            )
            created.append(complaint)

        # Resolve two of the four -> resolution rate must be exactly 50%.
        for complaint in created[:2]:
            await manager.update(
                complaint.id, ComplaintUpdate(status="resolved"), actor="tester"
            )
        await seeded_session.commit()

        overview = await StatisticsService(seeded_session).overview()
        assert overview.kpis.total_complaints == 4
        assert overview.kpis.resolved_complaints == 2
        assert overview.kpis.resolution_rate == 0.5

    async def test_frequency_distribution_percentages_sum_to_100(
        self, seeded_session
    ) -> None:
        from app.schemas.complaint import ComplaintCreate
        from app.services.ai.pipeline import AIPipeline
        from app.services.complaint_manager import ComplaintManager
        from tests.conftest import StubAnalyzer

        pipeline = AIPipeline(
            ml_analyzer=StubAnalyzer(),  # type: ignore[arg-type]
            llm_analyzer=StubAnalyzer(available=False),  # type: ignore[arg-type]
            use_llm_for_summary=False,
        )
        manager = ComplaintManager(seeded_session, pipeline)
        for index in range(3):
            await manager.create(
                ComplaintCreate(
                    description=f"Drain is blocked near the market area number {index}.",
                    location="Market Street",
                ),
                check_duplicates=False,
            )
        await seeded_session.commit()

        distribution = await StatisticsService(seeded_session).frequency_distribution(
            "category"
        )
        assert distribution.total == 3
        assert sum(item.percentage for item in distribution.items) == pytest.approx(100.0)
        assert distribution.mode_label is not None
