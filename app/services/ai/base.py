"""The AI contract.

``AIAnalyzer`` is the abstraction the Batch 5 (OOP) benchmark asks for: a clear class
that owns AI operations and sits inside the workflow rather than beside it. Everything
downstream — ``ComplaintManager``, the API, the UI — depends only on this interface, so
swapping the scikit-learn implementation for the Claude one (or adding a third) requires
no change anywhere else.

Two concrete implementations ship:

* ``MLAnalyzer``  — scikit-learn, local, free, always available. The default.
* ``LLMAnalyzer`` — Claude, optional, better summaries. Enabled only when a key is set.

``AIPipeline`` composes them and guarantees a result even when both fail.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from app.models.enums import (
    CATEGORY_TO_DEPARTMENT_SLUG,
    AIEngine,
    ComplaintCategory,
    ComplaintPriority,
)


@dataclass
class Prediction:
    """One candidate label with its confidence — used for the runner-up list."""

    label: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "confidence": round(self.confidence, 4)}


@dataclass
class AIResult:
    """Everything the AI layer produces for one complaint.

    This is persisted verbatim into ``Complaint.ai_output``, so the UI can explain a
    prediction ("Drainage, 87% — next best Water, 9%") and an auditor can see which
    engine and model version produced it long after a retrain.
    """

    category: ComplaintCategory
    priority: ComplaintPriority
    summary: str
    engine: AIEngine
    model_version: str

    category_confidence: float = 0.0
    priority_confidence: float = 0.0
    category_alternatives: list[Prediction] = field(default_factory=list)
    priority_alternatives: list[Prediction] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    recommended_department: str | None = None
    matched_terms: list[str] = field(default_factory=list)
    processing_ms: float = 0.0
    #: Human-readable trace of what happened — "LLM timed out, used ML result".
    #: Surfaced in the admin UI so an operator can see when the AI was degraded.
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.recommended_department is None:
            self.recommended_department = CATEGORY_TO_DEPARTMENT_SLUG.get(
                self.category, "general-administration"
            )

    @property
    def is_confident(self) -> bool:
        """Below this the UI nudges an administrator to review the classification."""
        return self.category_confidence >= 0.55

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["category"] = self.category.value
        payload["priority"] = self.priority.value
        payload["engine"] = self.engine.value
        payload["category_alternatives"] = [
            prediction.to_dict() for prediction in self.category_alternatives
        ]
        payload["priority_alternatives"] = [
            prediction.to_dict() for prediction in self.priority_alternatives
        ]
        payload["category_confidence"] = round(self.category_confidence, 4)
        payload["priority_confidence"] = round(self.priority_confidence, 4)
        payload["processing_ms"] = round(self.processing_ms, 2)
        return payload


class AIAnalyzer(ABC):
    """Contract every AI engine must satisfy."""

    #: Short identifier used in logs and in the ``engine`` field of the result.
    name: str = "abstract"

    @property
    @abstractmethod
    def available(self) -> bool:
        """Whether this engine can currently serve a request.

        Checked before every call: the ML analyzer reports ``False`` when its artifacts
        are missing, the LLM analyzer when no API key is configured.
        """

    @abstractmethod
    async def analyze(self, description: str, location: str | None = None) -> AIResult:
        """Turn an unstructured complaint into structured, actionable fields.

        Implementations should raise ``AIServiceError`` on failure rather than returning
        a degraded result — ``AIPipeline`` owns the fallback decision, not the engine.
        """

    def describe(self) -> dict[str, Any]:
        """Introspection for the ``/ai/status`` endpoint."""
        return {"name": self.name, "available": self.available}
