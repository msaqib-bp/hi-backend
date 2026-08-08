"""The scikit-learn analyzer — the default engine.

Loads the trained artifacts once at import time and serves every prediction locally: no
network, no API key, no per-request cost. This is what makes the deployed demo free to
run and impossible to break with an expired credential.

Inference runs in a worker thread (``anyio.to_thread``). scikit-learn is synchronous and
CPU-bound; calling it directly from an async handler would block the event loop and stall
every other request for the duration.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import anyio
import joblib

from app.core.config import settings
from app.core.exceptions import AIServiceError
from app.core.logging import get_logger
from app.ml.constants import (
    CATEGORY_MODEL_FILE,
    METADATA_FILE,
    PRIORITY_MODEL_FILE,
    SIMILARITY_MODEL_FILE,
)
from app.ml.lexicon import blend_scores, matched_terms
from app.ml.preprocess import (
    URGENCY_TERMS,
    clean_text,
    extract_keywords,
    split_sentences,
    tidy_sentence,
)
from app.models.enums import AIEngine, ComplaintCategory, ComplaintPriority
from app.services.ai.base import AIAnalyzer, AIResult, Prediction

log = get_logger(__name__)


class MLAnalyzer(AIAnalyzer):
    """TF-IDF + linear models, blended with a curated domain lexicon."""

    name = "ml"

    def __init__(self, artifact_dir: Path | None = None) -> None:
        self._artifact_dir = artifact_dir or settings.ML_ARTIFACT_DIR
        self._category_model: Any = None
        self._priority_model: Any = None
        self._similarity_vectorizer: Any = None
        self._metadata: dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        """Load artifacts from disk. Missing artifacts are not fatal — the pipeline
        falls back to keyword rules, so the API still accepts complaints."""
        try:
            category_path = self._artifact_dir / CATEGORY_MODEL_FILE
            priority_path = self._artifact_dir / PRIORITY_MODEL_FILE
            similarity_path = self._artifact_dir / SIMILARITY_MODEL_FILE

            if not category_path.exists() or not priority_path.exists():
                log.warning(
                    "ml_artifacts_missing",
                    artifact_dir=str(self._artifact_dir),
                    hint="Run: python -m app.ml.train",
                )
                return

            self._category_model = joblib.load(category_path)
            self._priority_model = joblib.load(priority_path)
            if similarity_path.exists():
                self._similarity_vectorizer = joblib.load(similarity_path)

            metadata_path = self._artifact_dir / METADATA_FILE
            if metadata_path.exists():
                self._metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

            log.info(
                "ml_artifacts_loaded",
                model_version=self._metadata.get("model_version"),
                trained_at=self._metadata.get("trained_at"),
            )
        except Exception as exc:  # pragma: no cover - corrupt artifact path
            log.error("ml_artifact_load_failed", error=str(exc), exc_info=True)
            self._category_model = None
            self._priority_model = None

    @property
    def available(self) -> bool:
        return self._category_model is not None and self._priority_model is not None

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    @property
    def similarity_vectorizer(self) -> Any:
        """Exposed for ``DuplicateDetector``, which reuses the fitted vocabulary."""
        return self._similarity_vectorizer

    # ---------------------------------------------------------------- inference
    async def analyze(self, description: str, location: str | None = None) -> AIResult:
        if not self.available:
            raise AIServiceError("The ML models are not loaded. Run `python -m app.ml.train`.")
        return await anyio.to_thread.run_sync(self._analyze_sync, description, location)

    def _analyze_sync(self, description: str, location: str | None) -> AIResult:
        started = time.perf_counter()
        cleaned = clean_text(description)

        category, category_confidence, category_alternatives, lexicon_used = (
            self._predict_category(description, cleaned)
        )
        priority, priority_confidence, priority_alternatives = self._predict_priority(description)

        notes: list[str] = []
        if lexicon_used:
            notes.append("Category blended with the curated civic lexicon.")
        if category_confidence < 0.55:
            notes.append("Low category confidence — worth a human review.")

        return AIResult(
            category=category,
            priority=priority,
            summary=self._summarize(description, category, priority, location),
            engine=AIEngine.ML,
            model_version=self._metadata.get("model_version", settings.MODEL_VERSION),
            category_confidence=category_confidence,
            priority_confidence=priority_confidence,
            category_alternatives=category_alternatives,
            priority_alternatives=priority_alternatives,
            keywords=extract_keywords(description),
            matched_terms=matched_terms(cleaned, category),
            processing_ms=(time.perf_counter() - started) * 1000,
            notes=notes,
        )

    def _predict_category(
        self, raw_text: str, cleaned: str
    ) -> tuple[ComplaintCategory, float, list[Prediction], bool]:
        probabilities = self._category_model.predict_proba([raw_text])[0]
        # sklearn hands back numpy scalars (np.str_ / np.float64). They serialise to
        # JSON in surprising ways, so cast to native types at the boundary.
        classes = [str(label) for label in self._category_model.classes_]
        model_scores = dict(zip(classes, (float(p) for p in probabilities), strict=True))

        blended, lexicon_used = blend_scores(model_scores, cleaned)
        ranked = sorted(blended.items(), key=lambda pair: pair[1], reverse=True)

        best_label, best_score = ranked[0]
        alternatives = [Prediction(label, score) for label, score in ranked[1:4]]
        return ComplaintCategory(best_label), best_score, alternatives, lexicon_used

    def _predict_priority(
        self, raw_text: str
    ) -> tuple[ComplaintPriority, float, list[Prediction]]:
        probabilities = self._priority_model.predict_proba([raw_text])[0]
        classes = [str(label) for label in self._priority_model.classes_]
        ranked = sorted(
            zip(classes, (float(p) for p in probabilities), strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        best_label, best_score = ranked[0]
        alternatives = [Prediction(label, score) for label, score in ranked[1:3]]
        return ComplaintPriority(best_label), best_score, alternatives

    # -------------------------------------------------------------- summarising
    def _summarize(
        self,
        description: str,
        category: ComplaintCategory,
        priority: ComplaintPriority,
        location: str | None,
    ) -> str:
        """Build a one-line dispatch summary without an LLM.

        Extractive, not generative: pick the most informative sentence from what the
        citizen wrote and prefix it with the routing facts a crew needs. It cannot
        hallucinate a detail the complaint never contained, which is the right
        trade-off for an operational instruction someone will act on.
        """
        sentences = split_sentences(description)
        core = max(enumerate(sentences), key=self._sentence_score)[1] if sentences else description
        core = tidy_sentence(core)

        where = f" at {location.strip()}" if location and location.strip() else ""
        return f"{priority.label} priority {category.label.lower()} issue{where}: {core}."

    @staticmethod
    def _sentence_score(index_sentence: tuple[int, str]) -> float:
        """Rank a sentence by how well it describes the actual problem.

        Three things this has to get right, learned from watching it fail:

        1. **Negated urgency.** "not urgent" contains an urgency term. Counting it would
           make the summary of "The bulb has fused. Not urgent." read *"Not urgent"* —
           discarding the entire problem statement. Urgency words preceded by a negator
           are ignored.
        2. **Fragments.** Short closers ("Thank you", "Please help") score well on
           density but say nothing. Sentences under four tokens are heavily penalised.
        3. **Position.** People lead with the problem and close with pleasantries, so
           the opening sentence gets a bias.
        """
        index, sentence = index_sentence
        tokens = clean_text(sentence).split()
        if not tokens:
            return -1.0

        negators = {"not", "no", "never", "nothing", "isnt", "arent", "dont"}
        urgency = 0
        for position, token in enumerate(tokens):
            if token not in URGENCY_TERMS:
                continue
            window = tokens[max(0, position - 2) : position]
            if not any(word in negators for word in window):
                urgency += 1

        length_score = min(len(tokens) / 18.0, 1.0)
        position_bonus = 0.5 if index == 0 else 0.0
        fragment_penalty = -1.5 if len(tokens) < 4 else 0.0
        return urgency * 0.8 + length_score + position_bonus + fragment_penalty

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "available": self.available,
            "model_version": self._metadata.get("model_version"),
            "trained_at": self._metadata.get("trained_at"),
            "training_samples": self._metadata.get("training_samples"),
            "category_macro_f1": self._metadata.get("category_macro_f1"),
            "priority_macro_f1": self._metadata.get("priority_macro_f1"),
            "generalization": self._metadata.get("generalization"),
        }
