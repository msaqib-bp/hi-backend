"""AI endpoints: engine introspection, re-analysis and the civic assistant."""

from __future__ import annotations

import uuid

from fastapi import APIRouter

from app.api.deps import AdminUser, ManagerDep, PipelineDep, StatsDep
from app.models.enums import ComplaintCategory, ComplaintPriority
from app.schemas.ai import AIStatus, AssistantRequest, AssistantResponse, ReanalyzeResponse

router = APIRouter(prefix="/ai", tags=["ai"])

#: Shown in the UI as starting points, so a judge does not have to invent a question.
SUGGESTED_QUESTIONS = [
    "Which department is slowest to resolve complaints?",
    "What type of problem is most common this month?",
    "Are critical complaints being handled faster than low priority ones?",
    "Is the backlog growing or shrinking?",
    "Which locations generate the most complaints?",
]


@router.get("/status", response_model=AIStatus, summary="Which AI engine is live")
async def ai_status(pipeline: PipelineDep) -> AIStatus:
    """Report the active engine, model version and measured accuracy.

    Exists for transparency: the spec requires every participant to explain what the AI
    is and what it does, and this endpoint answers that live rather than from a slide.
    """
    info = pipeline.describe()
    generalization = info.get("generalization") or {}

    notes = []
    if info["ml_available"]:
        notes.append("scikit-learn classifiers loaded and serving predictions locally.")
    else:
        notes.append("ML artifacts missing — run `python -m app.ml.train`.")

    if info["llm_available"]:
        notes.append(
            f"Language model enabled via {info['llm_provider']} — writing dispatch "
            "summaries and answering assistant questions. Category and priority still "
            "come from the local classifiers, which are calibrated and measurable."
        )
    else:
        notes.append(
            "No language model configured (set ANTHROPIC_API_KEY or DEEPSEEK_API_KEY). "
            "Summaries are extractive and the assistant returns a statistics digest — "
            "every other capability is unaffected."
        )

    if generalization:
        notes.append(
            f"Category accuracy on unseen phrasings: "
            f"{generalization.get('category_accuracy', 0):.1%} "
            f"(the honest number; the random-split score is inflated by shared phrasings)."
        )

    return AIStatus(
        ml_available=info["ml_available"],
        llm_available=info["llm_available"],
        llm_provider=info["llm_provider"],
        active_engine=info["active_engine"],
        model_version=info["model_version"],
        categories=[category.value for category in ComplaintCategory],
        priorities=[priority.value for priority in ComplaintPriority],
        trained_at=info.get("trained_at"),
        training_samples=info.get("training_samples"),
        # Report the unseen-phrasing F1, not the random-split one. The random split
        # shares issue phrasings between train and test and scores ~1.0, which would be
        # a misleading headline number to publish on a status endpoint.
        macro_f1=generalization.get("category_macro_f1") or info.get("macro_f1"),
        notes=notes,
    )


@router.post(
    "/complaints/{complaint_id}/reanalyze",
    response_model=ReanalyzeResponse,
    summary="Re-run the AI over an existing complaint",
)
async def reanalyze(
    complaint_id: uuid.UUID, manager: ManagerDep, _: AdminUser
) -> ReanalyzeResponse:
    """Useful after a model retrain, or for a second opinion on a doubtful case.

    Human overrides are preserved — re-analysis will not silently undo a correction an
    administrator made deliberately.
    """
    complaint, previous, current = await manager.reanalyze(complaint_id)
    return ReanalyzeResponse(
        reference_code=complaint.reference_code,
        previous=previous,
        current=current,
        changed=bool(current.get("changed")),
    )


@router.post("/assistant", response_model=AssistantResponse, summary="Ask about the data")
async def assistant(
    payload: AssistantRequest, pipeline: PipelineDep, stats: StatsDep
) -> AssistantResponse:
    """Answer a natural-language question about live complaint data.

    The statistics are computed first and passed to the model as context, so the answer
    is grounded in real aggregates rather than invented. The same context is returned in
    the response, which means any claim in the answer can be checked against the numbers
    the model was actually given.

    With no API key this degrades to a deterministic digest of those same statistics —
    less conversational, still accurate.
    """
    context = await stats.assistant_context()
    answer, engine = await pipeline.answer_question(payload.question, context)

    return AssistantResponse(
        answer=answer,
        engine=engine,
        grounded_on=context,
        suggestions=SUGGESTED_QUESTIONS,
    )
