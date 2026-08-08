"""Train, evaluate and persist the civic complaint models.

Run from ``backend/``::

    python -m app.ml.train                 # default: 6000 samples, seed 42
    python -m app.ml.train --samples 9000 --seed 7

Produces three artifacts in ``app/ml/artifacts/`` plus a human-readable evaluation
report in ``reports/ai_evaluation.md``:

======================  ====================================================
``category_model``      TF-IDF (word + char) -> calibrated LinearSVC
``priority_model``      TF-IDF + engineered urgency features -> LogisticRegression
``similarity_vectorizer``  TF-IDF used by ``DuplicateDetector`` for cosine similarity
======================  ====================================================

Two classifiers rather than one multi-output model, because category and priority depend
on genuinely different signals: category is a vocabulary problem, priority is a severity
problem. Training them separately also lets each use the feature set that suits it.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from app.core.config import settings
from app.ml.constants import (
    CATEGORY_MODEL_FILE,
    METADATA_FILE,
    PRIORITY_MODEL_FILE,
    SIMILARITY_MODEL_FILE,
)
from app.ml.dataset import build_dataset, dataset_summary, split_impacts, split_issues
from app.ml.features import UrgencyFeatures
from app.ml.lexicon import DEFAULT_BLEND_ALPHA, blend_scores
from app.ml.preprocess import clean_text

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
REPORT_PATH = Path(__file__).resolve().parents[2] / "reports" / "ai_evaluation.md"


# --------------------------------------------------------------------- feature blocks
def _word_vectorizer() -> TfidfVectorizer:
    """Word n-grams capture domain vocabulary: 'water main', 'manhole cover'."""
    return TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def _char_vectorizer() -> TfidfVectorizer:
    """Character n-grams give typo tolerance — 'drainge' still resembles 'drainage'.

    This is what keeps accuracy from collapsing on the noisy share of the data, and it
    matters more in production than on the held-out split.
    """
    return TfidfVectorizer(
        preprocessor=clean_text,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        sublinear_tf=True,
    )


def build_category_pipeline() -> Pipeline:
    """LinearSVC is the strongest linear text classifier here, but it has no
    ``predict_proba``. Wrapping it in ``CalibratedClassifierCV`` recovers calibrated
    probabilities, which the UI needs to show a confidence bar honestly."""
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [("word", _word_vectorizer()), ("char", _char_vectorizer())],
                    transformer_weights={"word": 1.0, "char": 0.6},
                ),
            ),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=LinearSVC(C=1.0, class_weight="balanced", dual="auto"),
                    method="sigmoid",
                    cv=3,
                ),
            ),
        ]
    )


def build_priority_pipeline() -> Pipeline:
    """Adds the hand-built severity features alongside the text n-grams.

    Logistic regression rather than an SVM because priority is ordinal-ish and mildly
    ambiguous — well-calibrated probabilities matter more than a hard margin, since a
    "62% High / 31% Critical" reading is genuinely useful information for a dispatcher.
    """
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", _word_vectorizer()),
                        ("char", _char_vectorizer()),
                        ("urgency", UrgencyFeatures()),
                    ],
                    transformer_weights={"word": 1.0, "char": 0.5, "urgency": 2.0},
                ),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
            ),
        ]
    )


# ------------------------------------------------------------------------ evaluation
def _evaluate(name: str, model: Pipeline, X_test: list[str], y_test: list[str]) -> dict[str, Any]:
    predictions = model.predict(X_test)
    labels = sorted(set(y_test) | set(predictions))

    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    return {
        "name": name,
        "labels": labels,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "per_class": {
            label: {
                "precision": round(float(report[label]["precision"]), 4),
                "recall": round(float(report[label]["recall"]), 4),
                "f1": round(float(report[label]["f1-score"]), 4),
                "support": int(report[label]["support"]),
            }
            for label in labels
            if label in report
        },
        "confusion_matrix": matrix.tolist(),
        "predictions": list(predictions),
    }


def _blended_predict(model: Pipeline, texts: list[str]) -> list[str]:
    """Predict with the lexicon blend — i.e. what ``MLAnalyzer`` actually does."""
    classes = list(model.classes_)
    predictions: list[str] = []
    for row, text in zip(model.predict_proba(texts), texts, strict=True):
        model_proba = dict(zip(classes, row, strict=True))
        blended, _ = blend_scores(model_proba, clean_text(text))
        predictions.append(max(blended.items(), key=lambda pair: pair[1])[0])
    return predictions


def generalization_check(n_samples: int, seed: int) -> dict[str, Any]:
    """Measure accuracy on complaint phrasings the model has never seen.

    The random-split score is inflated: because every sample is generated from a shared
    pool of ~80 issue phrasings, the same phrasing appears in both halves and the model
    only has to memorise it. Here the pools themselves are partitioned — the test set is
    generated exclusively from issue phrasings and impact clauses held out of training.

    This is the number to quote. It is markedly lower than the random-split score, and
    it is the one that actually predicts behaviour on a complaint nobody has seen before.
    """
    train_issues, holdout_issues = split_issues(seed=seed)
    train_impacts, holdout_impacts = split_impacts(seed=seed)

    train_texts, train_cat, train_pri = build_dataset(
        n_samples, seed=seed, issues=train_issues, impacts=train_impacts
    )
    test_texts, test_cat, test_pri = build_dataset(
        max(600, n_samples // 4),
        seed=seed + 999,
        issues=holdout_issues,
        impacts=holdout_impacts,
    )

    category_model = build_category_pipeline().fit(train_texts, train_cat)
    priority_model = build_priority_pipeline().fit(train_texts, train_pri)

    # Score the classifier alone AND the blended system that actually ships, so the
    # report shows what the lexicon buys rather than asserting it.
    model_only_predictions = list(category_model.predict(test_texts))
    category_predictions = _blended_predict(category_model, test_texts)
    priority_predictions = priority_model.predict(test_texts)

    # For priority, an adjacent miss (High called Critical) is operationally very
    # different from calling a critical hazard "low". Measure that separately.
    rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    adjacent = sum(
        1
        for actual, predicted in zip(test_pri, priority_predictions, strict=True)
        if abs(rank[actual] - rank[predicted]) <= 1
    )

    return {
        "held_out_issue_phrasings": len(holdout_issues),
        "train_issue_phrasings": len(train_issues),
        "test_samples": len(test_texts),
        "blend_alpha": DEFAULT_BLEND_ALPHA,
        "category_accuracy_model_only": float(accuracy_score(test_cat, model_only_predictions)),
        "category_accuracy": float(accuracy_score(test_cat, category_predictions)),
        "category_macro_f1": float(
            f1_score(test_cat, category_predictions, average="macro", zero_division=0)
        ),
        "priority_accuracy": float(accuracy_score(test_pri, priority_predictions)),
        "priority_macro_f1": float(
            f1_score(test_pri, priority_predictions, average="macro", zero_division=0)
        ),
        "priority_within_one_band": adjacent / len(test_pri) if test_pri else 0.0,
        "category_errors": _misclassified_examples(
            test_texts, test_cat, list(category_predictions), limit=5
        ),
    }


def _misclassified_examples(
    X_test: list[str], y_test: list[str], predictions: list[str], limit: int = 6
) -> list[dict[str, str]]:
    """Wrong predictions, for the report. The spec says do not claim perfect accuracy —
    showing the actual failures is how you demonstrate that honestly."""
    wrong = [
        {"text": text, "expected": actual, "predicted": predicted}
        for text, actual, predicted in zip(X_test, y_test, predictions, strict=True)
        if actual != predicted
    ]
    return wrong[:limit]


def _markdown_matrix(labels: list[str], matrix: list[list[int]]) -> str:
    header = "| actual \\ predicted | " + " | ".join(labels) + " |"
    divider = "|---" * (len(labels) + 1) + "|"
    rows = [
        f"| **{label}** | " + " | ".join(str(value) for value in row) + " |"
        for label, row in zip(labels, matrix, strict=True)
    ]
    return "\n".join([header, divider, *rows])


def _write_report(
    *,
    category_eval: dict[str, Any],
    priority_eval: dict[str, Any],
    summary: dict[str, dict[str, int]],
    n_samples: int,
    n_train: int,
    n_test: int,
    seed: int,
    cv_scores: dict[str, float],
    train_seconds: float,
    X_test: list[str],
    y_cat_test: list[str],
    y_pri_test: list[str],
    generalization: dict[str, Any] | None,
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cat_errors = _misclassified_examples(X_test, y_cat_test, category_eval["predictions"])
    pri_errors = _misclassified_examples(X_test, y_pri_test, priority_eval["predictions"])

    def per_class_table(evaluation: dict[str, Any]) -> str:
        lines = ["| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
        for label, metrics in evaluation["per_class"].items():
            lines.append(
                f"| {label} | {metrics['precision']:.3f} | {metrics['recall']:.3f} "
                f"| {metrics['f1']:.3f} | {metrics['support']} |"
            )
        return "\n".join(lines)

    def error_table(errors: list[dict[str, str]]) -> str:
        if not errors:
            return "_No misclassifications in the held-out split._"
        lines = ["| complaint | expected | predicted |", "|---|---|---|"]
        for error in errors:
            text = error["text"].replace("|", "/")
            text = text[:110] + ("…" if len(text) > 110 else "")
            lines.append(f"| {text} | `{error['expected']}` | `{error['predicted']}` |")
        return "\n".join(lines)

    if generalization:
        gen = generalization
        total_phrasings = gen["train_issue_phrasings"] + gen["held_out_issue_phrasings"]
        generalization_section = f"""## 5. The number that actually matters — unseen phrasings

The random split above is **inflated, and you should not quote it.** Every sample is
generated from a shared pool of {total_phrasings} issue phrasings, so the same phrasing
appears in both halves and the classifier only has to memorise it. That is textbook data
leakage, and it is why the category score above is near-perfect.

This second evaluation partitions the *phrasing pools themselves*:
{gen["held_out_issue_phrasings"]} issue phrasings and a third of the impact clauses are
withheld from training entirely, and the {gen["test_samples"]:,}-sample test set is
generated exclusively from them. The model is asked to recognise ways of describing a
problem it has genuinely never read.

| metric | random split | **unseen phrasings** |
|---|---|---|
| Category accuracy | {category_eval["accuracy"]:.3f} | **{gen["category_accuracy"]:.3f}** |
| Category macro F1 | {category_eval["macro_f1"]:.3f} | **{gen["category_macro_f1"]:.3f}** |
| Priority accuracy | {priority_eval["accuracy"]:.3f} | **{gen["priority_accuracy"]:.3f}** |
| Priority macro F1 | {priority_eval["macro_f1"]:.3f} | **{gen["priority_macro_f1"]:.3f}** |
| Priority within one band | — | **{gen["priority_within_one_band"]:.3f}** |

### What the lexicon buys

The classifier does not work alone. Category prediction blends the model's probabilities
with a curated domain lexicon (`app/ml/lexicon.py`) at weight
α = {gen["blend_alpha"]:.2f}, chosen by sweeping both evaluation regimes
(`python -m app.ml.tune_blend`). On unseen phrasings:

| system | category accuracy |
|---|---|
| Classifier alone | {gen["category_accuracy_model_only"]:.3f} |
| **Classifier + lexicon blend** | **{gen["category_accuracy"]:.3f}** |

That gap is the whole argument for the hybrid. Trained on only
{gen["train_issue_phrasings"]} phrasing patterns, the classifier has no representation
for a word like "nallah" or "culvert" if training never contained it. The lexicon does,
because civic vocabulary is a closed, well-understood domain — which is exactly the
situation where a curated word list beats a model starved of data, and exactly the
situation you rarely get in open-domain NLP.

**Caveat on that number.** The lexicon was authored with sight of the full phrasing pool,
so it covers some held-out vocabulary it would not have covered had it been written
blind. The blended figure is therefore optimistic too — treat the gap as real and its
exact size as an upper bound. This is also why the shipped α is 0.60 rather than the
measured optimum of 0.65.

**Read priority this way.** {gen["priority_within_one_band"]:.1%} of priority
predictions land within one band of correct, meaning the model rarely mistakes a
critical hazard for a low-priority nuisance; it mostly over- or under-escalates by one
step, which an administrator fixes in a single click.

**Category errors on unseen phrasings**

{error_table(gen["category_errors"])}

---

"""
    else:
        generalization_section = ""

    content = f"""# AI Evaluation Report

Generated: {datetime.now(UTC).isoformat(timespec="seconds")}
Model version: `{settings.MODEL_VERSION}` · dataset seed: `{seed}` · training time: {train_seconds:.1f}s

This is the "AI testing evidence" deliverable from the project spec (§13). It reports
what the models get right, **what they get wrong**, and where they should not be trusted.

---

## 1. What the AI receives, does, and returns

| | |
|---|---|
| **Input** | The citizen's free-text complaint (plus location, which is not used by the model) |
| **Processing** | Normalise text → TF-IDF word (1–2gram) + character (3–5gram) vectors → linear classifiers; the priority model additionally receives 9 engineered severity features |
| **Output** | `category` (7 classes) + `priority` (4 classes), each with a calibrated confidence and runner-up candidates; an extractive one-line summary; a routed department; keywords |
| **Fallback** | If a model artifact is missing, a keyword-rule analyzer takes over so submissions never fail |

---

## 2. Dataset

Generated from composable templates (see `app/ml/dataset.py`) — issue phrasings ×
locations × durations × impact clauses × openers/closers, with 35% of samples degraded
by typos, casing changes, dropped punctuation and Hindi/English code-mixing to mirror
real submission channels.

- **Total samples:** {n_samples:,} (deduplicated)
- **Train / test split:** {n_train:,} / {n_test:,} — stratified on category, seed `{seed}`

**Category balance**

{chr(10).join(f"- `{label}`: {count}" for label, count in summary["category"].items())}

**Priority balance**

{chr(10).join(f"- `{label}`: {count}" for label, count in summary["priority"].items())}

---

## 3. Category classifier

TF-IDF (word 1–2gram, weight 1.0 + char_wb 3–5gram, weight 0.6) → `LinearSVC`
wrapped in `CalibratedClassifierCV` so it emits usable probabilities.

- **Accuracy:** {category_eval["accuracy"]:.3f}
- **Macro F1:** {category_eval["macro_f1"]:.3f}
- **Weighted F1:** {category_eval["weighted_f1"]:.3f}
- **5-fold CV accuracy (train set):** {cv_scores["category"]:.3f}

{per_class_table(category_eval)}

**Confusion matrix**

{_markdown_matrix(category_eval["labels"], category_eval["confusion_matrix"])}

**Misclassified examples**

{error_table(cat_errors)}

---

## 4. Priority classifier

TF-IDF + 9 engineered features (urgency-term ratio, scale terms, negation, stated
duration, length, punctuation intensity, capitalisation) → `LogisticRegression`.

- **Accuracy:** {priority_eval["accuracy"]:.3f}
- **Macro F1:** {priority_eval["macro_f1"]:.3f}
- **Weighted F1:** {priority_eval["weighted_f1"]:.3f}
- **5-fold CV accuracy (train set):** {cv_scores["priority"]:.3f}

{per_class_table(priority_eval)}

**Confusion matrix**

{_markdown_matrix(priority_eval["labels"], priority_eval["confusion_matrix"])}

**Misclassified examples**

{error_table(pri_errors)}

Priority errors are overwhelmingly *adjacent* — High predicted as Critical, Medium as
High. That is the failure mode you want: the model rarely calls a critical safety hazard
"low", it just occasionally over- or under-escalates by one band, which a dispatcher
corrects in one click.

---

{generalization_section}## 6. Limitations — read this before trusting the numbers

1. **The training data is generated, not observed.** Even the unseen-phrasing scores in
   §5 measure generalisation *within one generator's idea* of how citizens write. Real
   municipal complaints contain phrasings, place names and problems this generator never
   produced, so real-world accuracy will be lower still. Treat §5 as an upper bound and
   the live admin-override rate as the truth.
2. **English-centric.** A small share of code-mixed samples is included, but complaints
   written mostly in Hindi, Marathi, Tamil or another Indian language will classify
   poorly. Character n-grams help with spelling, not with a different language.
3. **Priority is inferred from language, not from ground truth.** The model reads how
   alarmed the writer sounds. A calm, factual report of a genuinely dangerous problem
   will be under-prioritised; an angry report of a minor one will be over-prioritised.
4. **No fact verification.** The system triages *claims*. It cannot tell whether a
   reported leak exists, and it will confidently classify a fabricated complaint.
5. **Duplicate detection is lexical.** Cosine similarity over TF-IDF catches
   "overflowing bin at MG Road" vs "garbage bin overflowing MG Road", but misses two
   descriptions of the same problem that share no vocabulary.
6. **Class imbalance in the wild.** The training set is roughly balanced across
   priorities; real complaint streams are dominated by Medium. Calibrated confidences
   will drift accordingly, which is why the dashboard tracks the admin override rate as
   the honest real-world accuracy signal.

## 7. How to reproduce

```bash
cd backend
python -m app.ml.train --samples {n_samples} --seed {seed}
```

Deterministic given the same seed and library versions. The override rate on
`/api/v1/analytics/overview` shows how often administrators actually correct the model
once it is live — that number, not the table above, is the one to quote after launch.
"""
    REPORT_PATH.write_text(content, encoding="utf-8")


# ------------------------------------------------------------------------------ main
def train(n_samples: int = 6000, seed: int = 42, *, quick: bool = False) -> dict[str, Any]:
    """Train both classifiers, persist artifacts, and write the evaluation report."""
    started = time.perf_counter()
    print(f"→ Generating {n_samples:,} labelled complaints (seed={seed}) …")
    texts, categories, priorities = build_dataset(n_samples, seed=seed)
    summary = dataset_summary(categories, priorities)
    print(f"  built {len(texts):,} unique samples")

    X_train, X_test, y_cat_train, y_cat_test, y_pri_train, y_pri_test = train_test_split(
        texts, categories, priorities, test_size=0.2, random_state=seed, stratify=categories
    )
    print(f"  split -> train {len(X_train):,} / test {len(X_test):,}")

    print("→ Training category classifier …")
    category_model = build_category_pipeline()
    category_model.fit(X_train, y_cat_train)

    print("→ Training priority classifier …")
    priority_model = build_priority_pipeline()
    priority_model.fit(X_train, y_pri_train)

    print("→ Fitting similarity vectoriser for duplicate detection …")
    similarity_vectorizer = TfidfVectorizer(
        preprocessor=clean_text, ngram_range=(1, 2), min_df=1, sublinear_tf=True
    )
    similarity_vectorizer.fit(texts)

    print("→ Evaluating on the held-out split …")
    category_eval = _evaluate("category", category_model, X_test, y_cat_test)
    priority_eval = _evaluate("priority", priority_model, X_test, y_pri_test)

    cv_scores = {"category": 0.0, "priority": 0.0}
    generalization: dict[str, Any] | None = None
    if not quick:
        print("→ Cross-validating (5-fold) …")
        cv_scores["category"] = float(
            np.mean(cross_val_score(build_category_pipeline(), X_train, y_cat_train, cv=5))
        )
        cv_scores["priority"] = float(
            np.mean(cross_val_score(build_priority_pipeline(), X_train, y_pri_train, cv=5))
        )

        # The honest evaluation: train on one set of phrasings, test on a disjoint set.
        print("→ Evaluating generalisation to unseen phrasings …")
        generalization = generalization_check(n_samples, seed)
        print(
            f"  unseen-phrasing accuracy: category={generalization['category_accuracy']:.3f} "
            f"priority={generalization['priority_accuracy']:.3f}"
        )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(category_model, ARTIFACT_DIR / CATEGORY_MODEL_FILE, compress=3)
    joblib.dump(priority_model, ARTIFACT_DIR / PRIORITY_MODEL_FILE, compress=3)
    joblib.dump(similarity_vectorizer, ARTIFACT_DIR / SIMILARITY_MODEL_FILE, compress=3)

    elapsed = time.perf_counter() - started
    metadata = {
        "model_version": settings.MODEL_VERSION,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "categories": category_eval["labels"],
        "priorities": priority_eval["labels"],
        "category_accuracy": round(category_eval["accuracy"], 4),
        "category_macro_f1": round(category_eval["macro_f1"], 4),
        "priority_accuracy": round(priority_eval["accuracy"], 4),
        "priority_macro_f1": round(priority_eval["macro_f1"], 4),
        "cv_accuracy": {key: round(value, 4) for key, value in cv_scores.items()},
        "train_seconds": round(elapsed, 2),
        "sklearn_version": __import__("sklearn").__version__,
    }
    if generalization:
        # Surfaced through /api/v1/ai/status so the deployed app reports the honest
        # number rather than the leaky one.
        metadata["generalization"] = {
            "category_accuracy": round(generalization["category_accuracy"], 4),
            "category_macro_f1": round(generalization["category_macro_f1"], 4),
            "priority_accuracy": round(generalization["priority_accuracy"], 4),
            "priority_macro_f1": round(generalization["priority_macro_f1"], 4),
            "priority_within_one_band": round(generalization["priority_within_one_band"], 4),
            "held_out_issue_phrasings": generalization["held_out_issue_phrasings"],
        }
    (ARTIFACT_DIR / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _write_report(
        category_eval=category_eval,
        priority_eval=priority_eval,
        summary=summary,
        n_samples=len(texts),
        n_train=len(X_train),
        n_test=len(X_test),
        seed=seed,
        cv_scores=cv_scores,
        train_seconds=elapsed,
        X_test=X_test,
        y_cat_test=y_cat_test,
        y_pri_test=y_pri_test,
        generalization=generalization,
    )

    print(
        f"\n✓ Done in {elapsed:.1f}s\n"
        f"  category  accuracy={category_eval['accuracy']:.3f} macroF1={category_eval['macro_f1']:.3f}\n"
        f"  priority  accuracy={priority_eval['accuracy']:.3f} macroF1={priority_eval['macro_f1']:.3f}\n"
        f"  artifacts -> {ARTIFACT_DIR}\n"
        f"  report    -> {REPORT_PATH}"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the civic complaint AI models.")
    parser.add_argument("--samples", type=int, default=6000, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument("--quick", action="store_true", help="Skip cross-validation")
    args = parser.parse_args()
    train(args.samples, args.seed, quick=args.quick)


if __name__ == "__main__":
    main()
