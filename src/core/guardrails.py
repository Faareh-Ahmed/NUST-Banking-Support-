"""
src/core/guardrails.py
-----------------------
Input and output safety filters for the NUST Bank AI support system.
Covers jailbreak detection, blocked-topic filtering, and PII scrubbing.
"""

import logging
import re
from dataclasses import dataclass
from typing import List

from src.core.settings import cfg

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """Immutable container returned by every guardrail check."""
    is_safe: bool
    reason: str = ""
    filtered_text: str = ""

    def __repr__(self) -> str:
        return f"GuardrailResult(safe={self.is_safe}, reason={self.reason!r})"


# ── Input Guardrails ──────────────────────────────────────────────────────────

def detect_jailbreak(text: str) -> GuardrailResult:
    """Return unsafe result if a prompt-injection pattern is detected."""
    text_lower = text.lower()
    for pattern in cfg.guardrails.jailbreak_patterns:
        if re.search(pattern, text_lower):
            logger.warning("Jailbreak attempt detected — pattern: %s", pattern)
            return GuardrailResult(
                is_safe=False,
                reason=(
                    "Your message appears to contain a prompt manipulation attempt. "
                    "I can only assist with NUST Bank related queries."
                ),
            )
    return GuardrailResult(is_safe=True)


def detect_blocked_topics(text: str) -> GuardrailResult:
    """Return unsafe result if the query touches a disallowed topic."""
    text_lower = text.lower()
    for topic in cfg.guardrails.blocked_topics:
        if topic in text_lower:
            logger.warning("Blocked topic detected: %s", topic)
            return GuardrailResult(
                is_safe=False,
                reason=(
                    f"I'm unable to provide information related to '{topic}'. "
                    "For security-sensitive requests, please contact NUST Bank "
                    "directly at +92 (51) 111 000 494."
                ),
            )
    return GuardrailResult(is_safe=True)


def check_input_safety(text: str) -> GuardrailResult:
    """
    Run all input guardrails in order.
    Returns the first failure encountered, or a safe result.
    """
    for check in (detect_jailbreak, detect_blocked_topics):
        result = check(text)
        if not result.is_safe:
            return result
    return GuardrailResult(is_safe=True)


# ── Output Guardrails ─────────────────────────────────────────────────────────

def scrub_pii(text: str) -> str:
    """Replace any PII that may have leaked into the model's response."""
    for _, pattern in cfg.guardrails.pii_patterns.items():
        text = re.sub(pattern, "[REDACTED]", text)
    return text


def check_output_safety(text: str) -> GuardrailResult:
    """
    Verify the generated response is safe to display.
    Scrubs PII and checks for patterns that disclose sensitive data.
    """
    cleaned = scrub_pii(text)

    for pattern in cfg.guardrails.harmful_output_patterns:
        if re.search(pattern, cleaned, re.IGNORECASE):
            logger.warning("Harmful content detected in model output.")
            return GuardrailResult(
                is_safe=False,
                reason=(
                    "The response was filtered because it contained potentially "
                    "sensitive information. Please contact NUST Bank for assistance."
                ),
                filtered_text="",
            )

    return GuardrailResult(is_safe=True, filtered_text=cleaned)


# ── Out-of-Domain Detection ───────────────────────────────────────────────────

def is_out_of_domain(query: str, retrieved_scores: List[float]) -> bool:
    """
    Decide whether a query is out-of-domain based on retrieval scores.

    If every retrieved chunk has a low cosine similarity the query is
    considered outside the bank's knowledge base.
    """
    if not retrieved_scores:
        return True
    avg_score = sum(retrieved_scores) / len(retrieved_scores)
    max_score = max(retrieved_scores)
    return (
        max_score < cfg.retriever.ood_max_score_threshold
        and avg_score < cfg.retriever.ood_avg_score_threshold
    )
