"""Verify the configured language-model provider actually works.

    python -m app.check_llm

Reads whatever is in the environment / ``.env``, reports which provider was selected and
why, then makes three real calls — triage, summary and an assistant question — printing
what came back. Run it after setting a key, so a bad key or an exhausted balance surfaces
here rather than silently degrading every complaint to the extractive summary.

Exit code 0 = the provider works (or none is configured, which is a valid state);
1 = a provider is configured but failing.
"""

from __future__ import annotations

import asyncio
import sys

from app.core.config import settings
from app.models.enums import ComplaintCategory
from app.services.ai.pipeline import build_llm_provider

SAMPLE = (
    "A burst water pipeline is flooding the main road near the hospital entrance and "
    "ambulances cannot pass. Please send someone immediately."
)


async def main() -> int:
    print("Configured LLM provider")
    print("-" * 60)
    print(f"  LLM_PROVIDER setting : {settings.LLM_PROVIDER}")
    print(f"  ANTHROPIC_API_KEY    : {'set' if settings.anthropic_configured else 'not set'}")
    print(
        f"  DEEPSEEK/compatible  : "
        f"{'set' if settings.openai_compatible_configured else 'not set'}"
    )
    print(f"  -> resolved provider : {settings.active_llm_provider}")

    provider = build_llm_provider()
    if not provider.available:
        print(
            "\nNo language model is configured.\n"
            "This is a valid, fully-functional state: the local scikit-learn models "
            "handle classification, priority, routing, summaries and duplicate "
            "detection.\nSet DEEPSEEK_API_KEY or ANTHROPIC_API_KEY to enable "
            "LLM-written summaries and the natural-language assistant."
        )
        return 0

    details = provider.describe()
    print(f"  provider label       : {details.get('provider')}")
    print(f"  model                : {details.get('model')}")
    if base_url := details.get("base_url"):
        print(f"  endpoint             : {base_url}")

    failures = 0

    print("\n1. Triage (classification + priority + summary)")
    print("-" * 60)
    try:
        result = await provider.analyze(SAMPLE, "Hospital Road, Ward 4")
        print(f"  category : {result.category.value}")
        print(f"  priority : {result.priority.value}")
        print(f"  summary  : {result.summary}")
        print(f"  latency  : {result.processing_ms:.0f} ms")
    except Exception as exc:
        failures += 1
        print(f"  FAILED: {exc}")

    print("\n2. Dispatch summary (the pipeline's default use of the LLM)")
    print("-" * 60)
    try:
        summary = await provider.summarize(SAMPLE, ComplaintCategory.WATER, "Hospital Road")
        print(f"  {summary}")
    except Exception as exc:
        failures += 1
        print(f"  FAILED: {exc}")

    print("\n3. Civic assistant (grounded on statistics)")
    print("-" * 60)
    try:
        answer = await provider.answer_question(
            "Which department is slowest?",
            {
                "kpis": {"total_complaints": 181, "open_complaints": 5, "resolution_rate": 0.88},
                "slowest_department": "Public Safety",
                "fastest_department": "Electrical",
            },
        )
        print("  " + answer.replace("\n", "\n  "))
    except Exception as exc:
        failures += 1
        print(f"  FAILED: {exc}")

    print("\n" + "=" * 60)
    if failures:
        print(
            f"{failures} of 3 checks failed. The app still works — every failure "
            "degrades to the local models — but the LLM features are not active."
        )
        return 1

    print("All checks passed. LLM features are live.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
