"""Engineered features for the priority model.

Category is almost purely a vocabulary problem, so TF-IDF alone handles it. Priority is
not: "there is a water leak" and "a burst water main is flooding the highway" share most
of their vocabulary but differ enormously in urgency. These hand-built features give the
priority classifier explicit signal about severity, scale and time pressure, which makes
it markedly more robust on phrasings the training data never contained.

This class is pickled into the model artifact, so it must stay importable at this path.
Renaming or moving it invalidates every saved model — retrain if you do.
"""

from __future__ import annotations

import re

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

from app.ml.preprocess import MINOR_TERMS, SCALE_TERMS, URGENCY_TERMS, clean_text

_DURATION_RE = re.compile(r"\b(\d+)\s*(hour|hours|day|days|week|weeks|month|months)\b")
_NEGATION_RE = re.compile(r"\b(no|not|never|without|zero)\b")

FEATURE_NAMES: tuple[str, ...] = (
    "urgency_term_ratio",
    "urgency_term_present",
    "minor_term_present",
    "scale_term_ratio",
    "negation_present",
    "duration_days",
    "token_count",
    "exclamation_count",
    "uppercase_ratio",
)


class UrgencyFeatures(BaseEstimator, TransformerMixin):
    """Turn raw complaint text into a small dense matrix of severity signals.

    Each feature is bounded to roughly [0, 1] so it sits on a comparable scale to the
    L2-normalised TF-IDF block it gets stacked beside — otherwise a raw token count of
    80 would swamp every TF-IDF weight in the linear model.
    """

    def fit(self, X, y=None):  # noqa: N803, ARG002 - sklearn API
        return self

    def transform(self, X):  # noqa: N803 - sklearn API
        rows = [self._features_for(text) for text in X]
        return sparse.csr_matrix(np.asarray(rows, dtype=np.float64))

    def get_feature_names_out(self, input_features=None):  # noqa: ARG002 - sklearn API
        return np.asarray(FEATURE_NAMES, dtype=object)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _features_for(raw_text: str) -> list[float]:
        raw_text = raw_text or ""
        cleaned = clean_text(raw_text)
        tokens = cleaned.split()
        token_count = len(tokens) or 1
        token_set = set(tokens)

        urgency_hits = len(token_set & URGENCY_TERMS)
        scale_hits = len(token_set & SCALE_TERMS)

        # Convert any stated duration to days, capped at 30 — beyond a month the exact
        # number stops changing how urgent the complaint is.
        duration_days = 0.0
        if match := _DURATION_RE.search(cleaned):
            amount, unit = int(match.group(1)), match.group(2)
            multiplier = {"hour": 1 / 24, "day": 1.0, "week": 7.0, "month": 30.0}[
                unit.rstrip("s")
            ]
            duration_days = min(amount * multiplier, 30.0) / 30.0

        letters = [char for char in raw_text if char.isalpha()]
        uppercase_ratio = (
            sum(1 for char in letters if char.isupper()) / len(letters) if letters else 0.0
        )

        return [
            min(urgency_hits / token_count * 10, 1.0),
            1.0 if urgency_hits else 0.0,
            1.0 if token_set & MINOR_TERMS else 0.0,
            min(scale_hits / token_count * 10, 1.0),
            1.0 if _NEGATION_RE.search(cleaned) else 0.0,
            duration_days,
            min(token_count / 60.0, 1.0),
            min(raw_text.count("!") / 3.0, 1.0),
            uppercase_ratio,
        ]
