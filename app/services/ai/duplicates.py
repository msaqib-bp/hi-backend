"""Duplicate complaint detection.

A single burst water main generates twenty reports from twenty neighbours. Without
detection the queue shows twenty jobs, the dashboard counts twenty incidents, and the
statistics are wrong in a way that misdirects budget.

The detector compares a new complaint against recent **open** complaints using cosine
similarity over TF-IDF vectors, with two non-textual signals folded in:

* **location match** — the same words in the location field raise the score, because
  identical text about a different street is a different problem;
* **recency** — an identical complaint from six weeks ago is a *recurrence* worth
  tracking separately, not a duplicate to merge.

Nothing is merged automatically. Candidates are surfaced to the citizen ("this may
already be reported") and to the administrator, who decides. Auto-merging on a lexical
match would silently discard genuine reports.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import anyio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.ml.preprocess import clean_text
from app.models.base import utcnow
from app.models.complaint import Complaint
from app.models.enums import ComplaintStatus

log = get_logger(__name__)


class DuplicateDetector:
    """Finds complaints that probably describe the same incident."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        threshold: float | None = None,
        lookback_days: int | None = None,
        max_candidates: int = 400,
    ) -> None:
        self._session = session
        self._threshold = threshold if threshold is not None else settings.DUPLICATE_SIMILARITY_THRESHOLD
        self._lookback_days = (
            lookback_days if lookback_days is not None else settings.DUPLICATE_LOOKBACK_DAYS
        )
        # Bounds the comparison cost. Vectorising every complaint ever filed would make
        # submission latency grow without limit as the dataset does.
        self._max_candidates = max_candidates

    async def find_duplicates(
        self,
        description: str,
        location: str,
        *,
        exclude_id: Any | None = None,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Return likely duplicates, most similar first.

        An empty list is returned on any failure — duplicate detection is an assistive
        feature and must never block a submission.
        """
        try:
            candidates = await self._recent_open_complaints(exclude_id)
            if not candidates:
                return []

            scored = await anyio.to_thread.run_sync(
                self._score_candidates, description, location, candidates
            )
            return scored[:limit]
        except Exception as exc:  # pragma: no cover - defensive
            log.warning("duplicate_detection_failed", error=str(exc))
            return []

    async def _recent_open_complaints(self, exclude_id: Any | None) -> list[Complaint]:
        """Only unresolved complaints inside the lookback window are candidates."""
        cutoff = utcnow() - timedelta(days=self._lookback_days)
        statement = (
            select(Complaint)
            .where(
                Complaint.created_at >= cutoff,
                Complaint.status.notin_(
                    [ComplaintStatus.RESOLVED, ComplaintStatus.REJECTED]
                ),
                Complaint.duplicate_of_id.is_(None),
            )
            .order_by(Complaint.created_at.desc())
            .limit(self._max_candidates)
        )
        if exclude_id is not None:
            statement = statement.where(Complaint.id != exclude_id)

        result = await self._session.execute(statement)
        return list(result.scalars().all())

    def _score_candidates(
        self, description: str, location: str, candidates: list[Complaint]
    ) -> list[dict[str, Any]]:
        """Vectorise and score. Runs in a worker thread — it is CPU-bound."""
        corpus = [clean_text(description)] + [clean_text(c.description) for c in candidates]

        # Fitted per call on the candidate window rather than reusing the training
        # vectoriser: the IDF weights should reflect *this* city's current complaints,
        # so a locally common word like a ward name is correctly discounted.
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        try:
            matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            # Raised when the corpus has no usable vocabulary (e.g. all stop words).
            return []

        similarities = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        new_location = clean_text(location)

        scored: list[dict[str, Any]] = []
        for candidate, text_similarity in zip(candidates, similarities, strict=True):
            score = float(text_similarity) * 0.75
            score += self._location_similarity(new_location, clean_text(candidate.location)) * 0.25
            score *= self._recency_weight(candidate)

            if score >= self._threshold:
                scored.append(
                    {
                        "id": candidate.id,
                        "reference_code": candidate.reference_code,
                        "description": candidate.description,
                        "similarity": round(min(score, 1.0), 3),
                        "status": candidate.status,
                        "created_at": candidate.created_at,
                    }
                )

        scored.sort(key=lambda item: item["similarity"], reverse=True)
        return scored

    @staticmethod
    def _location_similarity(left: str, right: str) -> float:
        """Jaccard overlap of location words — cheap and adequate for free-text areas."""
        left_tokens, right_tokens = set(left.split()), set(right.split())
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def _recency_weight(self, candidate: Complaint) -> float:
        """Decay from 1.0 (today) to 0.6 at the edge of the lookback window.

        A near-identical complaint filed a month ago is more likely a recurring failure
        that deserves its own record than a duplicate of today's report.
        """
        age_days = (utcnow() - candidate.created_at).total_seconds() / 86400.0
        if age_days <= 2:
            return 1.0
        fraction = min(age_days / max(self._lookback_days, 1), 1.0)
        return 1.0 - 0.4 * fraction
