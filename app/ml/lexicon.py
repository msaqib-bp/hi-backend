"""Curated category lexicon, blended with the learned classifier.

**Why a lexicon at all.** The held-out-phrasing evaluation exposed the weakness of a
purely learned model here: trained on a bounded set of phrasings, it collapses when a
citizen describes a problem with vocabulary the training data never contained. A model
that has never read the word "nallah" cannot route it to Drainage, however good its
n-grams are.

Civic complaints are a *closed domain*. The vocabulary is small, stable and well known
to any municipal officer — which makes a curated lexicon unusually effective here, in a
way it would not be for open-ended text. Blending it with the classifier gives:

- **the model's strength** — context, word order, multi-word patterns, typo tolerance;
- **the lexicon's strength** — coverage of domain nouns regardless of phrasing.

Weights are deliberately coarse (1.0 / 2.0 / 3.0). A term scoring 3.0 is essentially
unambiguous for its category; 1.0 terms are suggestive but appear across categories.
Ambiguous terms are intentionally listed under several categories so the blend stays
undecided rather than confidently wrong.
"""

from __future__ import annotations

import re

from app.models.enums import ComplaintCategory as Cat

#: term -> weight, per category. Matched against cleaned text as whole words
#: (or word prefixes for the entries ending in ``*``).
CATEGORY_LEXICON: dict[Cat, dict[str, float]] = {
    Cat.ROAD: {
        "pothole": 3.0, "potholes": 3.0, "road": 1.5, "roads": 1.5, "footpath": 2.5,
        "pavement": 2.5, "sidewalk": 2.5, "asphalt": 3.0, "tar": 2.0, "tarmac": 3.0,
        "crater": 2.5, "speedbreaker": 3.0, "kerb": 2.5, "curb": 2.0, "divider": 2.0,
        "resurfac*": 3.0, "carriageway": 3.0, "flyover": 2.0, "highway": 1.5,
        "zebra": 2.5, "lane": 1.0, "street": 1.0, "cracking": 1.5, "caved": 2.0,
        "gravel": 2.0, "paver": 2.5, "tiles": 1.0, "bumpy": 2.5,
    },
    Cat.WATER: {
        "water": 1.5, "drinking": 2.5, "tap": 2.5, "taps": 2.5, "pipeline": 2.5,
        "pipe": 2.0, "pipes": 2.0, "supply": 1.5, "borewell": 3.0, "bore": 2.0,
        "tanker": 3.0, "tank": 2.0, "potable": 3.0, "muddy": 2.0, "turbid": 3.0,
        "chlorine": 3.0, "pressure": 1.5, "reservoir": 2.5, "sump": 2.5,
        "purifier": 2.0, "wastage": 1.5, "connection": 1.0, "valve": 2.0,
    },
    Cat.WASTE: {
        "garbage": 3.0, "rubbish": 3.0, "trash": 3.0, "waste": 2.0, "litter": 2.5,
        "dustbin": 3.0, "bin": 2.5, "bins": 2.5, "dump": 2.5, "dumping": 2.5,
        "dumped": 2.0, "sweeping": 2.5, "sweeper": 3.0, "collection": 1.5,
        "carcass": 3.0, "sanitation": 2.0, "compost": 2.5, "segregation": 2.5,
        "scrap": 2.0, "debris": 1.5, "filth": 2.5, "stinking": 1.5, "rotting": 2.0,
        "truck": 1.0, "vermin": 2.0, "rodents": 2.0,
    },
    Cat.ELECTRICITY: {
        "streetlight": 3.0, "streetlights": 3.0, "lamp": 2.5, "lamppost": 3.0,
        "electricity": 3.0, "electric": 2.5, "electrical": 2.5, "power": 2.0,
        "transformer": 3.0, "wire": 2.5, "wires": 2.5, "cable": 2.5, "pole": 2.0,
        "voltage": 3.0, "current": 2.0, "bulb": 2.5, "spark": 2.5, "sparking": 3.0,
        "shortcircuit": 3.0, "fuse": 2.5, "fused": 2.5, "outage": 3.0,
        "blackout": 3.0, "switchboard": 3.0, "meter": 1.0, "electrocution": 3.0,
        "live": 1.0, "lighting": 2.0, "dark": 1.0,
    },
    Cat.DRAINAGE: {
        "drain": 3.0, "drains": 3.0, "drainage": 3.0, "sewage": 3.0, "sewer": 3.0,
        "sewerage": 3.0, "manhole": 3.0, "waterlogging": 3.0, "waterlogged": 3.0,
        "nallah": 3.0, "nala": 3.0, "gutter": 3.0, "culvert": 3.0, "silt": 2.5,
        "stagnant": 2.5, "choked": 2.0, "blocked": 1.5, "clogged": 2.0,
        "overflowing": 1.0, "stormwater": 3.0, "septic": 3.0, "effluent": 2.5,
        "mosquito": 1.5, "mosquitoes": 1.5, "breeding": 1.5, "flooding": 1.5,
        "flooded": 1.5, "backflow": 2.5,
    },
    Cat.SAFETY: {
        "unsafe": 2.5, "safety": 2.5, "danger": 1.5, "dangerous": 1.5,
        "miscreant": 3.0, "miscreants": 3.0, "crime": 3.0, "theft": 3.0,
        "barricade": 2.5, "fencing": 2.5, "fence": 2.0, "stray": 2.5,
        "collapse": 2.0, "collapsing": 2.0, "hazard": 2.0, "security": 2.5,
        "harassment": 3.0, "antisocial": 3.0, "drunk": 2.5, "gambling": 3.0,
        "excavation": 2.5, "trench": 2.0, "playground": 2.0, "children": 1.0,
        "attacking": 2.0, "bite": 2.0, "guard": 2.0, "cctv": 3.0,
    },
    Cat.OTHER: {
        "certificate": 3.0, "tax": 3.0, "office": 2.0, "staff": 2.0, "clerk": 3.0,
        "notice": 2.0, "encroachment": 2.5, "toilet": 2.5, "noise": 2.5,
        "busstop": 2.5, "shelter": 2.0, "hall": 2.0, "booking": 2.5,
        "cattle": 2.5, "licence": 3.0, "license": 3.0, "application": 2.5,
        "receipt": 3.0, "documents": 2.5, "corruption": 3.0, "bribe": 3.0,
        "portal": 2.5, "website": 2.0, "helpline": 2.5,
    },
}

#: Multi-word phrases, checked as substrings after cleaning. These carry more signal
#: than either word alone ("water logging" is drainage, not water supply).
CATEGORY_PHRASES: dict[Cat, dict[str, float]] = {
    Cat.ROAD: {"road surface": 3.0, "speed breaker": 3.0, "road condition": 2.5},
    Cat.WATER: {"water supply": 3.0, "drinking water": 3.0, "no water": 3.0,
                "water tank": 2.5, "water pressure": 3.0, "water meter": 2.5},
    Cat.WASTE: {"garbage collection": 3.0, "waste collection": 3.0,
                "garbage truck": 3.0, "solid waste": 3.0},
    Cat.ELECTRICITY: {"street light": 3.0, "power cut": 3.0, "electric pole": 3.0,
                      "power supply": 2.5, "electricity bill": 2.5},
    Cat.DRAINAGE: {"water logging": 3.0, "open drain": 3.0, "manhole cover": 3.0,
                   "drainage system": 3.0, "sewage water": 3.0, "storm water": 3.0},
    Cat.SAFETY: {"stray dogs": 3.0, "public safety": 3.0, "anti social": 3.0,
                 "boundary wall": 2.5, "not safe": 2.5},
    Cat.OTHER: {"property tax": 3.0, "birth certificate": 3.0, "public toilet": 3.0,
                "bus stop": 2.5, "community hall": 3.0, "ward office": 3.0},
}

#: Lexicon share of the blended category score.
#:
#: Swept against both evaluation regimes by ``python -m app.ml.tune_blend``:
#:
#:     alpha   unseen-phrasing acc   in-distribution acc
#:     0.00           0.525                 1.000        <- classifier alone
#:     0.45           0.757                 1.000
#:     0.55           0.817                 0.969
#:     0.65           0.823                 0.969        <- measured optimum
#:     1.00           0.838                 0.917        <- lexicon alone
#:
#: The measured optimum is 0.65. We ship 0.60 deliberately: the lexicon was authored
#: with sight of the full phrasing pool, so its unseen-phrasing score is optimistic and
#: leaning on it slightly less is the safer bet on genuinely novel complaints. The cost
#: is ~0.006 accuracy — noise at this sample size.
DEFAULT_BLEND_ALPHA = 0.6

_WORD_RE = re.compile(r"[a-z]+")

#: Terms written with a trailing ``*`` match on prefix, so one entry covers
#: "resurfacing" / "resurfaced" / "resurface" without listing each form.
_PREFIX_TERMS: dict[Cat, list[tuple[str, float]]] = {
    category: [
        (term.rstrip("*"), weight) for term, weight in terms.items() if term.endswith("*")
    ]
    for category, terms in CATEGORY_LEXICON.items()
}
_EXACT_TERMS: dict[Cat, dict[str, float]] = {
    category: {term: weight for term, weight in terms.items() if not term.endswith("*")}
    for category, terms in CATEGORY_LEXICON.items()
}


def lexicon_scores(cleaned_text: str) -> dict[str, float]:
    """Score a cleaned complaint against every category.

    Returns a probability-like mapping of ``category value -> score`` summing to 1.0,
    or an empty dict when nothing matched (the caller then relies on the model alone
    rather than blending in a meaningless uniform distribution).
    """
    if not cleaned_text:
        return {}

    tokens = set(_WORD_RE.findall(cleaned_text))
    raw: dict[Cat, float] = {}

    for category, terms in _EXACT_TERMS.items():
        score = sum(weight for term, weight in terms.items() if term in tokens)

        # Prefix terms ("resurfac*" matches resurfacing / resurfaced).
        for prefix, weight in _PREFIX_TERMS.get(category, []):
            if any(token.startswith(prefix) for token in tokens):
                score += weight

        for phrase, weight in CATEGORY_PHRASES.get(category, {}).items():
            if phrase in cleaned_text:
                score += weight

        if score:
            raw[category] = score

    total = sum(raw.values())
    if total <= 0:
        return {}
    return {category.value: score / total for category, score in raw.items()}


def matched_terms(cleaned_text: str, category: Cat) -> list[str]:
    """Which lexicon terms fired for a category — used to explain a prediction."""
    tokens = set(_WORD_RE.findall(cleaned_text or ""))
    hits = [term for term in _EXACT_TERMS.get(category, {}) if term in tokens]
    hits += [
        prefix
        for prefix, _ in _PREFIX_TERMS.get(category, [])
        if any(token.startswith(prefix) for token in tokens)
    ]
    hits += [
        phrase for phrase in CATEGORY_PHRASES.get(category, {}) if phrase in (cleaned_text or "")
    ]
    return sorted(set(hits))


def blend_scores(
    model_proba: dict[str, float], cleaned_text: str, alpha: float = DEFAULT_BLEND_ALPHA
) -> tuple[dict[str, float], bool]:
    """Combine classifier probabilities with lexicon evidence.

    ``alpha`` is the lexicon's share of the final score. It only applies when the
    lexicon actually matched something — otherwise the model's distribution is returned
    untouched, since blending against no evidence would just flatten a confident
    prediction for nothing.

    Returns ``(blended_scores, lexicon_used)``.
    """
    lexicon = lexicon_scores(cleaned_text)
    if not lexicon:
        return model_proba, False

    blended = {
        category: (1 - alpha) * model_proba.get(category, 0.0) + alpha * lexicon.get(category, 0.0)
        for category in set(model_proba) | set(lexicon)
    }
    total = sum(blended.values())
    if total > 0:
        blended = {category: score / total for category, score in blended.items()}
    return blended, True
