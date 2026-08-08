"""Text preprocessing shared by training and inference.

This module is the single source of truth for how complaint text is normalised. If
training and inference clean text differently the model silently degrades in production,
so both paths import ``clean_text`` from here — never reimplement it.

The cleaning is deliberately light. Aggressive stemming or spell-correction destroys the
signal that separates "water leaking slowly" from "water main burst", and the character
n-gram half of the vectoriser already absorbs most typos.
"""

from __future__ import annotations

import re
import unicodedata

# Precompiled once: these run on every complaint submission.
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
_PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s-]{7,}\d)\b")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")
_WHITESPACE_RE = re.compile(r"\s+")
_NON_TEXT_RE = re.compile(r"[^a-z0-9\s'/-]")

#: Domain shorthand that citizens actually type. Expanding these before vectorising
#: gives the model one consistent token instead of a long tail of near-synonyms.
_ABBREVIATIONS: dict[str, str] = {
    "st light": "streetlight",
    "st lite": "streetlight",
    "street lite": "streetlight",
    "street light": "streetlight",
    "man hole": "manhole",
    "man-hole": "manhole",
    "water logging": "waterlogging",
    "water-logging": "waterlogging",
    "garbge": "garbage",
    "rubish": "rubbish",
    "elec": "electricity",
    "electric": "electricity",
    "corp": "corporation",
    "govt": "government",
    "rd": "road",
    "blk": "block",
    "opp": "opposite",
    "nr": "near",
    "pls": "please",
    "plz": "please",
    "sewrage": "sewerage",
    "drainge": "drainage",
    "potholes": "pothole",
}

#: Words that carry urgency regardless of category. Used by the engineered features in
#: ``app.ml.features`` and by the extractive summariser.
URGENCY_TERMS: frozenset[str] = frozenset(
    {
        "accident",
        "collapse",
        "collapsed",
        "danger",
        "dangerous",
        "dead",
        "death",
        "electrocution",
        "emergency",
        "explode",
        "exposed",
        "fatal",
        "fire",
        "flood",
        "flooded",
        "flooding",
        "hazard",
        "immediately",
        "injured",
        "injury",
        "life",
        "live",
        "overflowing",
        "risk",
        "severe",
        "shock",
        "sparking",
        "spreading",
        "unsafe",
        "urgent",
        "urgently",
    }
)

#: Words signalling a minor, non-urgent issue.
MINOR_TERMS: frozenset[str] = frozenset(
    {
        "cosmetic",
        "faded",
        "minor",
        "occasionally",
        "slight",
        "slightly",
        "small",
        "sometimes",
        "whenever",
    }
)

#: Terms that indicate the problem affects many people rather than one household.
SCALE_TERMS: frozenset[str] = frozenset(
    {
        "area",
        "colony",
        "entire",
        "everyone",
        "families",
        "highway",
        "hospital",
        "junction",
        "main",
        "market",
        "neighbourhood",
        "residents",
        "school",
        "society",
        "street",
        "traffic",
        "ward",
        "whole",
    }
)

#: Stop words tuned for this domain. The generic English list would strip "no" and
#: "not", which flip the meaning of "no water supply".
DOMAIN_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "our",
        "please",
        "sir",
        "that",
        "the",
        "their",
        "there",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "will",
        "with",
        "you",
        "your",
    }
)


def clean_text(text: str) -> str:
    """Normalise a complaint for the vectoriser.

    Steps: unicode-normalise, lowercase, strip URLs/emails/phone numbers, expand domain
    abbreviations, squash character runs (``pleaseeee`` -> ``pleasee``), drop stray
    punctuation, collapse whitespace.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text).lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _PHONE_RE.sub(" ", text)

    for abbreviation, expansion in _ABBREVIATIONS.items():
        text = text.replace(abbreviation, expansion)

    text = _REPEATED_CHAR_RE.sub(r"\1\1", text)
    text = _NON_TEXT_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Cleaned tokens with domain stop words removed."""
    return [tok for tok in clean_text(text).split() if tok not in DOMAIN_STOPWORDS and len(tok) > 1]


def extract_keywords(text: str, limit: int = 6) -> list[str]:
    """Salient terms for the AI output card.

    Frequency-ranked with urgency and scale words promoted, because those are what a
    dispatcher scans for. Deliberately not TF-IDF: this must work on a single document
    with no corpus available.
    """
    tokens = tokenize(text)
    if not tokens:
        return []

    scores: dict[str, float] = {}
    for token in tokens:
        weight = 1.0
        if token in URGENCY_TERMS:
            weight = 3.0
        elif token in SCALE_TERMS:
            weight = 2.0
        elif len(token) > 6:  # longer words tend to be the domain nouns
            weight = 1.5
        scores[token] = scores.get(token, 0.0) + weight

    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    return [token for token, _ in ranked[:limit]]


def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter — good enough for extractive summarisation."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [part.strip() for part in parts if part.strip()]
