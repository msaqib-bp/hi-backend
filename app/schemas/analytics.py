"""Analytics contracts.

Every response carries an ``interpretation`` string. The Batch 4 benchmark says plainly:
"Explain what the statistics mean rather than displaying numbers only" — so the
explanation is produced by the backend from the numbers themselves, not written by hand
in the UI.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class DescriptiveStats(BaseModel):
    """The full descriptive-statistics set required by the Statistics benchmark."""

    count: int
    mean: float | None = None
    median: float | None = None
    mode: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    range: float | None = None
    variance: float | None = None
    std_deviation: float | None = None
    interpretation: str


class QuartileStats(BaseModel):
    """Quartiles and Tukey fences, used to isolate abnormally slow complaints."""

    q1: float | None = None
    q2_median: float | None = None
    q3: float | None = None
    iqr: float | None = None
    lower_fence: float | None = None
    upper_fence: float | None = None
    outlier_count: int = 0
    outlier_references: list[str] = Field(default_factory=list)
    interpretation: str


class FrequencyItem(BaseModel):
    key: str
    label: str
    count: int
    percentage: float


class FrequencyDistribution(BaseModel):
    dimension: str
    total: int
    items: list[FrequencyItem]
    mode_label: str | None = None
    interpretation: str


class TrendPoint(BaseModel):
    period: date
    submitted: int
    resolved: int
    moving_average_7d: float | None = None


class TrendSeries(BaseModel):
    points: list[TrendPoint]
    total_submitted: int
    total_resolved: int
    direction: str = Field(description="rising | falling | steady")
    change_percent: float | None = None
    interpretation: str


class DepartmentPerformance(BaseModel):
    department: str
    slug: str
    total: int
    resolved: int
    open: int
    resolution_rate: float
    mean_resolution_hours: float | None = None
    median_resolution_hours: float | None = None
    overdue_count: int


class DepartmentPerformanceReport(BaseModel):
    departments: list[DepartmentPerformance]
    fastest: str | None = None
    slowest: str | None = None
    interpretation: str


class ResolutionTimeReport(BaseModel):
    """Everything about how long things take, in one payload."""

    descriptive: DescriptiveStats
    quartiles: QuartileStats
    by_priority: dict[str, DescriptiveStats]
    sla_breach_rate: float
    interpretation: str


class OverviewKPIs(BaseModel):
    total_complaints: int
    open_complaints: int
    in_progress: int
    resolved_complaints: int
    resolution_rate: float
    critical_open: int
    overdue_open: int
    submitted_last_7_days: int
    resolved_last_7_days: int
    mean_resolution_hours: float | None = None
    median_resolution_hours: float | None = None
    ai_override_rate: float = Field(
        description="Share of complaints where an admin corrected the AI — our honest "
        "real-world accuracy signal."
    )


class AnalyticsOverview(BaseModel):
    kpis: OverviewKPIs
    by_category: FrequencyDistribution
    by_priority: FrequencyDistribution
    by_status: FrequencyDistribution
    headline: str
    interpretation: str
