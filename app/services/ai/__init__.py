"""AI service layer."""

from app.services.ai.base import AIAnalyzer, AIResult, Prediction
from app.services.ai.duplicates import DuplicateDetector
from app.services.ai.llm_analyzer import LLMAnalyzer
from app.services.ai.llm_shared import LLMProvider, NullLLM
from app.services.ai.ml_analyzer import MLAnalyzer
from app.services.ai.openai_compat_analyzer import OpenAICompatibleAnalyzer
from app.services.ai.pipeline import AIPipeline, build_llm_provider, get_ai_pipeline
from app.services.ai.rule_analyzer import RuleAnalyzer

__all__ = [
    "AIAnalyzer",
    "AIPipeline",
    "AIResult",
    "DuplicateDetector",
    "LLMAnalyzer",
    "LLMProvider",
    "MLAnalyzer",
    "NullLLM",
    "OpenAICompatibleAnalyzer",
    "Prediction",
    "RuleAnalyzer",
    "build_llm_provider",
    "get_ai_pipeline",
]
