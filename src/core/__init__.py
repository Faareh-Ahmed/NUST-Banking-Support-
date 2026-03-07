"""Core RAG components: settings, guardrails, prompt engine, LLM engine."""

from src.core.settings import cfg
from src.core.guardrails import check_input_safety, check_output_safety, is_out_of_domain, GuardrailResult
from src.core.prompt_engine import build_rag_prompt, OUT_OF_DOMAIN_RESPONSE, SAFETY_RESPONSE
from src.core.llm_engine import LLMEngine

__all__ = [
    "cfg",
    "check_input_safety",
    "check_output_safety",
    "is_out_of_domain",
    "GuardrailResult",
    "build_rag_prompt",
    "OUT_OF_DOMAIN_RESPONSE",
    "SAFETY_RESPONSE",
    "LLMEngine",
]
