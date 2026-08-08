"""``StatisticsService`` — the analytics engine behind the dashboard.

This class implements the Batch 4 (Statistics) benchmark: central tendency, dispersion,
frequency distributions, quartiles with Tukey fences, and trends — each returned with a
generated **interpretation sentence**, because the benchmark says explicitly "explain
what the statistics mean rather than displaying numbers only".

Two implementation notes worth knowing:

*Why compute in Python rather than SQL.* Variance, quartiles and Tukey fences are either
non-portable or absent across SQLite and PostgreSQL, and the volumes here are small
(thousands of rows). Pulling the resolution-time vector into NumPy keeps one correct
implementation instead of two dialect-specific ones. Counting queries stay in SQL.

*Why every method guards on sample size.* An empty database, or one with three resolved
complaints, must not produce a confident-looking zero. Quartiles over n < 4 are
meaningless, so the service says so rather than returning a number that would be quoted.
"""

from __future__ import annotations

import statistics as stats
from collections import Counter
from datetime import timedelta
from typing import Any

import numpy as np
from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.base import utcnow
from app.models.complaint import Complaint
from app.models.department import Department
from app.models.enums import ComplaintCategory, ComplaintPriority, ComplaintStatus
from app.schemas.analytics import (
    AnalyticsOverview,
    DepartmentPerformance,
    DepartmentPerformanceReport,
    DescriptiveStats,
    FrequencyDistribution,
    FrequencyItem,
    OverviewKPIs,
    QuartileStats,
    ResolutionTimeReport,
    TrendPoint,
    TrendSeries,
)

#: Below this, quartiles and fences are not reported — with three points a "quartile"
#: is an artefact of the interpolation method, not a property of the data.
MIN_SAMPLES_FOR_QUARTILES = 4

#: Below this, dispersion statistics are unstable enough to be misleading.
MIN_SAMPLES_FOR_VARIANCE = 2


class StatisticsService:
    """Computes every statistic the dashboard displays."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ============================================================== descriptive
    @staticmethod
    def describe(values: list[float], unit: str = "hours") -> DescriptiveStats:
        """Full descriptive-statistics set for a numeric sample.

        Mode deserves a note: on continuous data every value is usually unique, so a raw
        mode is meaningless. Values are binned to the nearest hour first, which turns
        mode into the genuinely useful "most common resolution duration" — and the
        interpretation says that is what happened, rather than quietly reporting a
        number nobody can reproduce.
        """
        count = len(values)
        if count == 0:
            return DescriptiveStats(
                count=0,
                interpretation="No resolved complaints yet, so timing statistics cannot be computed.",
            )

        mean = float(np.mean(values))
        median = float(np.median(values))
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        value_range = maximum - minimum

        binned = [round(value) for value in values]
        mode_value = float(Counter(binned).most_common(1)[0][0])

        if count >= MIN_SAMPLES_FOR_VARIANCE:
            variance = float(stats.variance(values))
            std_deviation = float(stats.stdev(values))
        else:
            variance = std_deviation = None

        return DescriptiveStats(
            count=count,
            mean=round(mean, 2),
            median=round(median, 2),
            mode=round(mode_value, 2),
            minimum=round(minimum, 2),
            maximum=round(maximum, 2),
            range=round(value_range, 2),
            variance=round(variance, 2) if variance is not None else None,
            std_deviation=round(std_deviation, 2) if std_deviation is not None else None,
            interpretation=StatisticsService._interpret_descriptive(
                count, mean, median, mode_value, std_deviation, minimum, maximum, unit
            ),
        )

    @staticmethod
    def _interpret_descriptive(
        count: int,
        mean: float,
        median: float,
        mode_value: float,
        std_deviation: float | None,
        minimum: float,
        maximum: float,
        unit: str,
    ) -> str:
        parts = [
            f"Across {count} resolved complaints the average resolution time is "
            f"{mean:.1f} {unit} and the median is {median:.1f} {unit}."
        ]

        # Mean-vs-median divergence is the single most useful thing to say about a
        # skewed operational distribution, and it is exactly what a raw table hides.
        if median > 0:
            skew_ratio = mean / median
            if skew_ratio > 1.3:
                parts.append(
                    f"The mean sits {skew_ratio:.1f}× above the median, so a minority of "
                    "very slow cases is dragging the average up — most complaints are "
                    "handled faster than the average suggests."
                )
            elif skew_ratio < 0.77:
                parts.append(
                    "The mean is below the median, meaning a cluster of very fast "
                    "resolutions is pulling the average down."
                )
            else:
                parts.append(
                    "Mean and median are close, so resolution times are fairly "
                    "symmetric with no dominant tail."
                )

        parts.append(
            f"The most common resolution duration (to the nearest hour) is "
            f"{mode_value:.0f} {unit}."
        )

        if std_deviation is not None and mean > 0:
            cv = std_deviation / mean
            consistency = (
                "very consistent" if cv < 0.3 else "moderately variable" if cv < 0.75
                else "highly inconsistent"
            )
            parts.append(
                f"The standard deviation is {std_deviation:.1f} {unit} "
                f"({cv:.0%} of the mean), which makes turnaround {consistency}."
            )

        parts.append(
            f"The fastest took {minimum:.1f} {unit} and the slowest {maximum:.1f} {unit}."
        )
        return " ".join(parts)

    # ================================================================ quartiles
    @staticmethod
    def quartiles(values: list[float], references: list[str] | None = None) -> QuartileStats:
        """Q1/Q2/Q3, IQR and Tukey fences, with the outliers named.

        The upper fence (Q3 + 1.5·IQR) is the operationally interesting number: those
        are the complaints that took abnormally long by the data's own standard, rather
        than by an arbitrary threshold someone picked.
        """
        count = len(values)
        if count < MIN_SAMPLES_FOR_QUARTILES:
            return QuartileStats(
                interpretation=(
                    f"Only {count} resolved complaint(s) so far — at least "
                    f"{MIN_SAMPLES_FOR_QUARTILES} are needed before quartiles mean anything."
                )
            )

        array = np.asarray(values, dtype=float)
        q1, q2, q3 = (float(np.percentile(array, p)) for p in (25, 50, 75))
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        outlier_indices = [
            index
            for index, value in enumerate(values)
            if value < lower_fence or value > upper_fence
        ]
        outlier_references = (
            [references[index] for index in outlier_indices if index < len(references)]
            if references
            else []
        )

        return QuartileStats(
            q1=round(q1, 2),
            q2_median=round(q2, 2),
            q3=round(q3, 2),
            iqr=round(iqr, 2),
            lower_fence=round(lower_fence, 2),
            upper_fence=round(upper_fence, 2),
            outlier_count=len(outlier_indices),
            outlier_references=outlier_references[:10],
            interpretation=StatisticsService._interpret_quartiles(
                q1, q2, q3, iqr, upper_fence, len(outlier_indices), count
            ),
        )

    @staticmethod
    def _interpret_quartiles(
        q1: float, q2: float, q3: float, iqr: float, upper_fence: float,
        outlier_count: int, total: int,
    ) -> str:
        parts = [
            f"A quarter of complaints are resolved within {q1:.1f} hours, half within "
            f"{q2:.1f} hours, and three quarters within {q3:.1f} hours.",
            f"The middle 50% therefore spans {iqr:.1f} hours (the interquartile range).",
        ]

        if outlier_count:
            share = outlier_count / total
            parts.append(
                f"{outlier_count} complaint(s) — {share:.1%} of the resolved total — fall "
                f"outside the Tukey fences, taking longer than {upper_fence:.1f} hours. "
                "These are statistical outliers rather than slow-but-normal cases, and "
                "are worth investigating individually: they usually indicate a job that "
                "stalled waiting on something, not one that was simply hard."
            )
        else:
            parts.append(
                f"No complaint exceeds the upper fence of {upper_fence:.1f} hours, so "
                "there are no abnormally stalled cases — the slow ones are slow within "
                "a normal spread."
            )
        return " ".join(parts)

    # =============================================================== frequencies
    async def frequency_distribution(self, dimension: str) -> FrequencyDistribution:
        """Counts and percentages across one categorical dimension."""
        column_map = {
            "category": Complaint.category,
            "priority": Complaint.priority,
            "status": Complaint.status,
            "location": Complaint.location,
        }
        label_map: dict[str, Any] = {
            "category": lambda key: ComplaintCategory(key).label,
            "priority": lambda key: ComplaintPriority(key).label,
            "status": lambda key: ComplaintStatus(key).label,
            "location": lambda key: key,
        }

        if dimension == "department":
            return await self._department_frequency()
        if dimension not in column_map:
            raise ValueError(f"Unsupported dimension '{dimension}'.")

        column = column_map[dimension]
        result = await self._session.execute(
            select(column, func.count()).group_by(column).order_by(func.count().desc())
        )
        rows = result.all()
        total = sum(count for _, count in rows)

        items: list[FrequencyItem] = []
        for key, count in rows:
            raw_key = key.value if hasattr(key, "value") else str(key)
            try:
                label = label_map[dimension](raw_key)
            except (ValueError, KeyError):
                label = raw_key
            items.append(
                FrequencyItem(
                    key=raw_key,
                    label=label,
                    count=count,
                    percentage=round(count / total * 100, 1) if total else 0.0,
                )
            )

        # `location` is free text with a long tail; showing 200 one-off entries is noise.
        if dimension == "location" and len(items) > 12:
            head, tail = items[:12], items[12:]
            tail_count = sum(item.count for item in tail)
            head.append(
                FrequencyItem(
                    key="__other__",
                    label=f"{len(tail)} other locations",
                    count=tail_count,
                    percentage=round(tail_count / total * 100, 1) if total else 0.0,
                )
            )
            items = head

        return FrequencyDistribution(
            dimension=dimension,
            total=total,
            items=items,
            mode_label=items[0].label if items else None,
            interpretation=self._interpret_frequency(dimension, items, total),
        )

    async def _department_frequency(self) -> FrequencyDistribution:
        result = await self._session.execute(
            select(Department.name, Department.slug, func.count(Complaint.id))
            .join(Complaint, Complaint.assigned_department_id == Department.id, isouter=True)
            .group_by(Department.id)
            .order_by(func.count(Complaint.id).desc())
        )
        rows = result.all()
        total = sum(count for _, _, count in rows)
        items = [
            FrequencyItem(
                key=slug,
                label=name,
                count=count,
                percentage=round(count / total * 100, 1) if total else 0.0,
            )
            for name, slug, count in rows
        ]
        return FrequencyDistribution(
            dimension="department",
            total=total,
            items=items,
            mode_label=items[0].label if items else None,
            interpretation=self._interpret_frequency("department", items, total),
        )

    @staticmethod
    def _interpret_frequency(
        dimension: str, items: list[FrequencyItem], total: int
    ) -> str:
        if not items or total == 0:
            return "There are no complaints to analyse yet."

        top = items[0]
        parts = [
            f"{top.label} is the most frequent {dimension}, accounting for "
            f"{top.count} of {total} complaints ({top.percentage:.1f}%)."
        ]

        if len(items) > 1:
            second = items[1]
            gap = top.percentage - second.percentage
            if gap > 20:
                parts.append(
                    f"It dominates — {second.label} is a distant second at "
                    f"{second.percentage:.1f}%, so effort concentrated here would have "
                    "the largest effect."
                )
            elif gap < 5:
                parts.append(
                    f"{second.label} is close behind at {second.percentage:.1f}%, so "
                    "demand is split rather than concentrated."
                )
            else:
                parts.append(f"{second.label} follows at {second.percentage:.1f}%.")

        # Concentration matters for staffing: three areas carrying 80% of demand is a
        # very different planning problem from demand spread evenly.
        top_three_share = sum(item.percentage for item in items[:3])
        if len(items) > 3 and top_three_share > 75:
            parts.append(
                f"The top three account for {top_three_share:.0f}% of all complaints — "
                "demand is highly concentrated."
            )

        if dimension == "priority":
            urgent = sum(
                item.count for item in items if item.key in ("high", "critical")
            )
            if urgent:
                parts.append(
                    f"{urgent} complaints ({urgent / total:.0%}) are High or Critical "
                    "priority and need attention within 24 hours."
                )
        return " ".join(parts)

    # =================================================================== trends
    async def trends(self, days: int = 30) -> TrendSeries:
        """Daily submitted/resolved counts with a 7-day moving average."""
        window_start = (utcnow() - timedelta(days=days)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        submitted_rows = await self._session.execute(
            select(Complaint.created_at).where(Complaint.created_at >= window_start)
        )
        resolved_rows = await self._session.execute(
            select(Complaint.resolved_at).where(Complaint.resolved_at.is_not(None))
        )

        # Bucketing in Python avoids SQLite/Postgres date-truncation dialect differences.
        submitted_by_day = Counter(value.date() for (value,) in submitted_rows.all())
        resolved_by_day = Counter(
            value.date() for (value,) in resolved_rows.all() if value >= window_start
        )

        start_date = window_start.date()
        today = utcnow().date()
        day_span = (today - start_date).days + 1
        all_days = [start_date + timedelta(days=offset) for offset in range(day_span)]

        points: list[TrendPoint] = []
        submitted_series: list[int] = []
        for day in all_days:
            submitted = submitted_by_day.get(day, 0)
            submitted_series.append(submitted)
            moving_average = (
                round(float(np.mean(submitted_series[-7:])), 2)
                if len(submitted_series) >= 7
                else None
            )
            points.append(
                TrendPoint(
                    period=day,
                    submitted=submitted,
                    resolved=resolved_by_day.get(day, 0),
                    moving_average_7d=moving_average,
                )
            )

        total_submitted = sum(point.submitted for point in points)
        total_resolved = sum(point.resolved for point in points)
        direction, change_percent = self._trend_direction(submitted_series)

        return TrendSeries(
            points=points,
            total_submitted=total_submitted,
            total_resolved=total_resolved,
            direction=direction,
            change_percent=change_percent,
            interpretation=self._interpret_trend(
                days, total_submitted, total_resolved, direction, change_percent
            ),
        )

    @staticmethod
    def _trend_direction(series: list[int]) -> tuple[str, float | None]:
        """Compare the last week against the week before it."""
        if len(series) < 14:
            return "steady", None
        recent = float(np.mean(series[-7:]))
        previous = float(np.mean(series[-14:-7]))
        if previous == 0:
            return ("rising", None) if recent > 0 else ("steady", None)

        change = (recent - previous) / previous * 100
        if change > 10:
            return "rising", round(change, 1)
        if change < -10:
            return "falling", round(change, 1)
        return "steady", round(change, 1)

    @staticmethod
    def _interpret_trend(
        days: int, submitted: int, resolved: int, direction: str, change_percent: float | None
    ) -> str:
        if submitted == 0:
            return f"No complaints were submitted in the last {days} days."

        parts = [
            f"{submitted} complaints were submitted over the last {days} days "
            f"(about {submitted / days:.1f} per day) and {resolved} were resolved."
        ]

        # Whether the backlog is growing or shrinking is the operationally decisive
        # fact, and it is not visible from either count on its own.
        backlog_delta = submitted - resolved
        if backlog_delta > 0:
            parts.append(
                f"That leaves a net {backlog_delta} added to the backlog — the team is "
                "closing fewer complaints than arrive."
            )
        elif backlog_delta < 0:
            parts.append(
                f"The team closed {abs(backlog_delta)} more than arrived, so the backlog "
                "is shrinking."
            )
        else:
            parts.append("Intake and resolution are exactly balanced.")

        if change_percent is not None and direction != "steady":
            parts.append(
                f"Submissions are {direction}: the last 7 days are "
                f"{abs(change_percent):.0f}% {'above' if direction == 'rising' else 'below'} "
                "the previous 7."
            )
        elif change_percent is not None:
            parts.append("Weekly volume is stable compared with the previous week.")
        return " ".join(parts)

    # ================================================== resolution-time report
    async def resolution_time_report(self) -> ResolutionTimeReport:
        """Everything about how long complaints take, including per-priority breakdown."""
        result = await self._session.execute(
            select(Complaint).where(Complaint.resolved_at.is_not(None))
        )
        resolved = list(result.scalars().all())

        durations = [
            hours for c in resolved if (hours := c.resolution_hours) is not None and hours >= 0
        ]
        references = [
            c.reference_code
            for c in resolved
            if (hours := c.resolution_hours) is not None and hours >= 0
        ]

        descriptive = self.describe(durations)
        quartile_stats = self.quartiles(durations, references)

        by_priority: dict[str, DescriptiveStats] = {}
        for priority in ComplaintPriority:
            subset = [
                hours
                for c in resolved
                if c.priority is priority
                and (hours := c.resolution_hours) is not None
                and hours >= 0
            ]
            if subset:
                by_priority[priority.value] = self.describe(subset)

        # SLA breach: resolved later than the priority's target. This is the number a
        # service manager is actually accountable for.
        breaches = sum(
            1
            for c in resolved
            if (hours := c.resolution_hours) is not None
            and hours > c.priority.target_resolution_hours
        )
        breach_rate = breaches / len(resolved) if resolved else 0.0

        return ResolutionTimeReport(
            descriptive=descriptive,
            quartiles=quartile_stats,
            by_priority=by_priority,
            sla_breach_rate=round(breach_rate, 4),
            interpretation=self._interpret_resolution(
                descriptive, quartile_stats, by_priority, breach_rate, breaches, len(resolved)
            ),
        )

    @staticmethod
    def _interpret_resolution(
        descriptive: DescriptiveStats,
        quartile_stats: QuartileStats,
        by_priority: dict[str, DescriptiveStats],
        breach_rate: float,
        breaches: int,
        total: int,
    ) -> str:
        if descriptive.count == 0:
            return "No complaints have been resolved yet, so timing cannot be assessed."

        parts = [
            f"{breaches} of {total} resolved complaints ({breach_rate:.1%}) missed the "
            "response target for their priority level."
        ]

        if breach_rate > 0.4:
            parts.append(
                "That is a substantial share — either the targets are unrealistic for "
                "current staffing, or high-priority work is not being triaged first."
            )
        elif breach_rate < 0.1:
            parts.append("Service targets are being met consistently.")

        # The check that matters: are urgent complaints actually handled faster?
        critical = by_priority.get("critical")
        low = by_priority.get("low")
        if critical and low and critical.median is not None and low.median is not None:
            if critical.median < low.median:
                parts.append(
                    f"Prioritisation is working as intended — Critical complaints are "
                    f"resolved in a median of {critical.median:.1f} hours versus "
                    f"{low.median:.1f} hours for Low priority."
                )
            else:
                parts.append(
                    f"Prioritisation is not working: Critical complaints take a median "
                    f"of {critical.median:.1f} hours, no faster than the "
                    f"{low.median:.1f} hours for Low priority ones. Urgent work is not "
                    "reaching the front of the queue."
                )

        if quartile_stats.outlier_count:
            parts.append(
                f"{quartile_stats.outlier_count} complaint(s) are statistical outliers "
                f"taking more than {quartile_stats.upper_fence:.0f} hours."
            )
        return " ".join(parts)

    # ============================================================== departments
    async def department_performance(self) -> DepartmentPerformanceReport:
        result = await self._session.execute(
            select(Department).where(Department.is_active.is_(True))
        )
        departments = list(result.scalars().all())

        performances: list[DepartmentPerformance] = []
        for department in departments:
            complaints_result = await self._session.execute(
                select(Complaint).where(Complaint.assigned_department_id == department.id)
            )
            complaints = list(complaints_result.scalars().all())
            if not complaints:
                continue

            resolved = [c for c in complaints if c.resolved_at is not None]
            durations = [
                hours
                for c in resolved
                if (hours := c.resolution_hours) is not None and hours >= 0
            ]

            performances.append(
                DepartmentPerformance(
                    department=department.name,
                    slug=department.slug,
                    total=len(complaints),
                    resolved=len(resolved),
                    open=len(complaints) - len(resolved),
                    resolution_rate=round(len(resolved) / len(complaints), 4),
                    mean_resolution_hours=round(float(np.mean(durations)), 2)
                    if durations
                    else None,
                    median_resolution_hours=round(float(np.median(durations)), 2)
                    if durations
                    else None,
                    overdue_count=sum(1 for c in complaints if c.is_overdue),
                )
            )

        performances.sort(key=lambda item: item.total, reverse=True)
        ranked = [item for item in performances if item.median_resolution_hours is not None]
        fastest = min(ranked, key=lambda item: item.median_resolution_hours) if ranked else None
        slowest = max(ranked, key=lambda item: item.median_resolution_hours) if ranked else None

        return DepartmentPerformanceReport(
            departments=performances,
            fastest=fastest.department if fastest else None,
            slowest=slowest.department if slowest else None,
            interpretation=self._interpret_departments(performances, fastest, slowest),
        )

    @staticmethod
    def _interpret_departments(
        performances: list[DepartmentPerformance],
        fastest: DepartmentPerformance | None,
        slowest: DepartmentPerformance | None,
    ) -> str:
        if not performances:
            return "No complaints have been assigned to a department yet."

        parts = [
            f"{performances[0].department} handles the largest share with "
            f"{performances[0].total} complaints."
        ]

        if fastest and slowest and fastest.slug != slowest.slug:
            parts.append(
                f"{fastest.department} is fastest at a median of "
                f"{fastest.median_resolution_hours:.1f} hours, while "
                f"{slowest.department} is slowest at "
                f"{slowest.median_resolution_hours:.1f} hours."
            )
            if (
                slowest.median_resolution_hours
                and fastest.median_resolution_hours
                and fastest.median_resolution_hours > 0
            ):
                ratio = slowest.median_resolution_hours / fastest.median_resolution_hours
                if ratio > 2:
                    parts.append(
                        f"That is a {ratio:.1f}× gap. A difference that large usually "
                        "reflects capacity or process, not complaint difficulty, and is "
                        "the clearest place to intervene."
                    )

        total_overdue = sum(item.overdue_count for item in performances)
        if total_overdue:
            worst = max(performances, key=lambda item: item.overdue_count)
            parts.append(
                f"{total_overdue} complaints are past their target time overall, "
                f"{worst.overdue_count} of them in {worst.department}."
            )
        return " ".join(parts)

    # ================================================================= overview
    async def overview(self) -> AnalyticsOverview:
        """Headline KPIs plus the three primary distributions."""
        total = (await self._session.execute(select(func.count(Complaint.id)))).scalar_one()

        async def count_where(*conditions) -> int:
            statement: Select = select(func.count(Complaint.id)).where(*conditions)
            return (await self._session.execute(statement)).scalar_one()

        open_count = await count_where(
            Complaint.status.in_([ComplaintStatus.OPEN, ComplaintStatus.ASSIGNED])
        )
        in_progress = await count_where(Complaint.status == ComplaintStatus.IN_PROGRESS)
        resolved_count = await count_where(Complaint.status == ComplaintStatus.RESOLVED)
        critical_open = await count_where(
            Complaint.priority == ComplaintPriority.CRITICAL,
            Complaint.status.notin_([ComplaintStatus.RESOLVED, ComplaintStatus.REJECTED]),
        )
        overridden = await count_where(Complaint.ai_overridden.is_(True))

        week_ago = utcnow() - timedelta(days=7)
        submitted_7d = await count_where(Complaint.created_at >= week_ago)
        resolved_7d = await count_where(Complaint.resolved_at >= week_ago)

        # `is_overdue` is clock-relative, so it is computed in Python over the open set.
        open_result = await self._session.execute(
            select(Complaint).where(
                Complaint.status.notin_([ComplaintStatus.RESOLVED, ComplaintStatus.REJECTED])
            )
        )
        overdue_open = sum(1 for c in open_result.scalars().all() if c.is_overdue)

        resolved_result = await self._session.execute(
            select(Complaint).where(Complaint.resolved_at.is_not(None))
        )
        durations = [
            hours
            for c in resolved_result.scalars().all()
            if (hours := c.resolution_hours) is not None and hours >= 0
        ]

        kpis = OverviewKPIs(
            total_complaints=total,
            open_complaints=open_count,
            in_progress=in_progress,
            resolved_complaints=resolved_count,
            resolution_rate=round(resolved_count / total, 4) if total else 0.0,
            critical_open=critical_open,
            overdue_open=overdue_open,
            submitted_last_7_days=submitted_7d,
            resolved_last_7_days=resolved_7d,
            mean_resolution_hours=round(float(np.mean(durations)), 2) if durations else None,
            median_resolution_hours=round(float(np.median(durations)), 2) if durations else None,
            ai_override_rate=round(overridden / total, 4) if total else 0.0,
        )

        by_category = await self.frequency_distribution("category")
        by_priority = await self.frequency_distribution("priority")
        by_status = await self.frequency_distribution("status")

        return AnalyticsOverview(
            kpis=kpis,
            by_category=by_category,
            by_priority=by_priority,
            by_status=by_status,
            headline=self._headline(kpis, by_category),
            interpretation=self._interpret_overview(kpis, by_category),
        )

    @staticmethod
    def _headline(kpis: OverviewKPIs, by_category: FrequencyDistribution) -> str:
        if kpis.total_complaints == 0:
            return "No complaints have been submitted yet."
        top = by_category.items[0].label if by_category.items else "unclassified issues"
        return (
            f"{kpis.total_complaints} complaints received, {kpis.resolution_rate:.0%} "
            f"resolved. {top} is the largest source of demand."
        )

    @staticmethod
    def _interpret_overview(kpis: OverviewKPIs, by_category: FrequencyDistribution) -> str:
        if kpis.total_complaints == 0:
            return (
                "The system is ready but no complaints have been submitted yet. "
                "Statistics will appear as reports arrive."
            )

        parts = [
            f"Of {kpis.total_complaints} complaints, {kpis.resolved_complaints} are "
            f"resolved ({kpis.resolution_rate:.1%}), {kpis.in_progress} are being worked "
            f"on and {kpis.open_complaints} are waiting."
        ]

        if kpis.critical_open:
            parts.append(
                f"⚠ {kpis.critical_open} Critical complaint(s) are still open and need "
                "attention within 6 hours."
            )
        if kpis.overdue_open:
            parts.append(
                f"{kpis.overdue_open} open complaint(s) have already passed the target "
                "time for their priority."
            )

        if kpis.median_resolution_hours is not None:
            parts.append(
                f"Typical resolution takes {kpis.median_resolution_hours:.1f} hours "
                f"(median), averaging {kpis.mean_resolution_hours:.1f} hours."
            )

        # The override rate is the honest accuracy signal — how often a human disagreed
        # with the model on real complaints, as opposed to a held-out test score.
        if kpis.total_complaints >= 20:
            parts.append(
                f"Administrators corrected the AI's classification on "
                f"{kpis.ai_override_rate:.1%} of complaints, which is the real-world "
                "accuracy signal for the model."
            )

        weekly_delta = kpis.submitted_last_7_days - kpis.resolved_last_7_days
        if weekly_delta > 0:
            parts.append(
                f"In the last 7 days {kpis.submitted_last_7_days} arrived but only "
                f"{kpis.resolved_last_7_days} were closed, so the backlog grew by "
                f"{weekly_delta}."
            )
        return " ".join(parts)

    # ========================================================= assistant context
    async def assistant_context(self) -> dict[str, Any]:
        """Compact snapshot of the live data, passed to the civic assistant.

        Returned to the caller alongside the answer so the response is auditable — you
        can see exactly which numbers the model was given.
        """
        overview = await self.overview()
        departments = await self.department_performance()
        resolution = await self.resolution_time_report()
        trend = await self.trends(days=30)

        return {
            "kpis": overview.kpis.model_dump(),
            "top_category": overview.by_category.mode_label,
            "category_breakdown": {
                item.label: item.count for item in overview.by_category.items
            },
            "priority_breakdown": {
                item.label: item.count for item in overview.by_priority.items
            },
            "status_breakdown": {item.label: item.count for item in overview.by_status.items},
            "resolution_time": {
                "mean_hours": resolution.descriptive.mean,
                "median_hours": resolution.descriptive.median,
                "std_deviation": resolution.descriptive.std_deviation,
                "q1": resolution.quartiles.q1,
                "q3": resolution.quartiles.q3,
                "sla_breach_rate": resolution.sla_breach_rate,
            },
            "fastest_department": departments.fastest,
            "slowest_department": departments.slowest,
            "departments": [
                {
                    "name": item.department,
                    "total": item.total,
                    "resolved": item.resolved,
                    "median_hours": item.median_resolution_hours,
                    "overdue": item.overdue_count,
                }
                for item in departments.departments
            ],
            "trend_30d": {
                "submitted": trend.total_submitted,
                "resolved": trend.total_resolved,
                "direction": trend.direction,
                "change_percent": trend.change_percent,
            },
        }
