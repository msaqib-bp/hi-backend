"""Last-resort keyword analyzer.

This engine exists for one scenario: the ML artifacts are missing or corrupt *and* there
is no API key. Without it, a fresh clone that has not run training would reject every
complaint submission — the worst possible failure, because it loses the citizen's report.

It reuses the same lexicon the ML engine blends with, so its category predictions are not
arbitrary; they are simply the lexicon acting alone, which the tuning sweep measured at
~0.82 accuracy on unseen phrasings. Priority falls back to counting urgency vocabulary.

Confidence is reported honestly (capped at 0.5) so the UI flags these for review.
"""

from __future__ import annotations

import time

from app.ml.lexicon import lexicon_scores, matched_terms
from app.ml.preprocess import (
    MINOR_TERMS,
    SCALE_TERMS,
    URGENCY_TERMS,
    clean_text,
    extract_keywords,
    split_sentences,
    tidy_sentence,
)
from app.models.enums import AIEngine, ComplaintCategory, ComplaintPriority
from app.services.ai.base import AIAnalyzer, AIResult, Prediction

#: Terms that on their own justify a Critical rating regardless of anything else.
_CRITICAL_MARKERS = frozenset(
    {
        "burst", "collapse", "collapsed", "electrocution", "explosion", "fatal",
        "fire", "live", "manhole", "sparking", "unconscious", "trapped",
    }
)


class RuleAnalyzer(AIAnalyzer):
    """Deterministic keyword analyzer — always available, never fails."""

    name = "fallback"

    @property
    def available(self) -> bool:
        return True

    async def analyze(self, description: str, location: str | None = None) -> AIResult:
        started = time.perf_counter()
        cleaned = clean_text(description)
        tokens = set(cleaned.split())

        category, category_confidence, alternatives = self._classify(cleaned)
        priority, priority_confidence = self._prioritize(tokens)

        sentences = split_sentences(description)
        core = tidy_sentence(sentences[0] if sentences else description)
        where = f" at {location.strip()}" if location and location.strip() else ""

        return AIResult(
            category=category,
            priority=priority,
            summary=f"{priority.label} priority {category.label.lower()} issue{where}: {core}.",
            engine=AIEngine.FALLBACK,
            model_version="rules-1.0",
            category_confidence=category_confidence,
            priority_confidence=priority_confidence,
            category_alternatives=alternatives,
            keywords=extract_keywords(description),
            matched_terms=matched_terms(cleaned, category),
            processing_ms=(time.perf_counter() - started) * 1000,
            notes=[
                "Keyword rules only — the trained models were unavailable. "
                "Please review this classification."
            ],
        )

    @staticmethod
    def _classify(cleaned: str) -> tuple[ComplaintCategory, float, list[Prediction]]:
        scores = lexicon_scores(cleaned)
        if not scores:
            return ComplaintCategory.OTHER, 0.15, []

        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        best_label, best_score = ranked[0]
        # Cap the reported confidence: this engine is a safety net, and overstating its
        # certainty would let a weak guess slip past review unflagged.
        return (
            ComplaintCategory(best_label),
            min(best_score, 0.5),
            [Prediction(label, min(score, 0.5)) for label, score in ranked[1:4]],
        )

    @staticmethod
    def _prioritize(tokens: set[str]) -> tuple[ComplaintPriority, float]:
        if tokens & _CRITICAL_MARKERS:
            return ComplaintPriority.CRITICAL, 0.5

        urgency_hits = len(tokens & URGENCY_TERMS)
        scale_hits = len(tokens & SCALE_TERMS)
        minor_hits = len(tokens & MINOR_TERMS)

        if urgency_hits >= 2:
            return ComplaintPriority.CRITICAL, 0.45
        if urgency_hits == 1 or scale_hits >= 2:
            return ComplaintPriority.HIGH, 0.4
        if minor_hits:
            return ComplaintPriority.LOW, 0.4
        return ComplaintPriority.MEDIUM, 0.35
