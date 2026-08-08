"""AI-facing contracts: engine status and the civic assistant."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AIStatus(BaseModel):
    """Introspection endpoint — lets the demo show exactly which engine is live."""

    model_config = ConfigDict(protected_namespaces=())

    ml_available: bool
    llm_available: bool
    #: Which vendor is serving the LLM features: "anthropic", "deepseek",
    #: "openai-compatible", or "none".
    llm_provider: str = "none"
    active_engine: str
    model_version: str
    categories: list[str]
    priorities: list[str]
    trained_at: str | None = None
    training_samples: int | None = None
    macro_f1: float | None = None
    notes: list[str] = Field(default_factory=list)


class AssistantRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        examples=["Which department has the slowest resolution time this month?"],
    )


class AssistantResponse(BaseModel):
    answer: str
    engine: str
    grounded_on: dict = Field(
        default_factory=dict,
        description="The live statistics passed to the model, so the answer is auditable.",
    )
    suggestions: list[str] = Field(default_factory=list)


class ReanalyzeResponse(BaseModel):
    reference_code: str
    previous: dict
    current: dict
    changed: bool
