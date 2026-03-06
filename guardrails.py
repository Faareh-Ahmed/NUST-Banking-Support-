"""
Guardrails & Safety Module
---------------------------
Content filtering, jailbreak detection, PII leak prevention,
and policy enforcement for the NUST Bank LLM system.
"""

import re
import logging
from typing import Tuple

from config import BLOCKED_TOPICS, JAILBREAK_PATTERNS, PII_PATTERNS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GuardrailResult:
    """Container for guardrail check results."""

    def __init__(self, is_safe: bool, reason: str = "", filtered_text: str = ""):
        self.is_safe = is_safe
        self.reason = reason
        self.filtered_text = filtered_text

    def __repr__(self):
        return f"GuardrailResult(safe={self.is_safe}, reason='{self.reason}')"


# ── Input Guardrails (applied to user query) ─────────────────────────────────

def detect_jailbreak(text: str) -> GuardrailResult:
    """Check for prompt injection / jailbreak attempts."""
    text_lower = text.lower()
    for pattern in JAILBREAK_PATTERNS:
        if re.search(pattern, text_lower):
            logger.warning("Jailbreak attempt detected: %s", pattern)
            return GuardrailResult(
                is_safe=False,
                reason="Your message appears to contain a prompt manipulation attempt. "
                       "I can only assist with NUST Bank related queries.",
            )
    return GuardrailResult(is_safe=True)


def detect_blocked_topics(text: str) -> GuardrailResult:
    """Check if the query requests sensitive / disallowed information."""
    text_lower = text.lower()
    for topic in BLOCKED_TOPICS:
        if topic in text_lower:
            logger.warning("Blocked topic detected: %s", topic)
            return GuardrailResult(
                is_safe=False,
                reason=f"I'm unable to provide information related to '{topic}'. "
                       "For security-sensitive requests, please contact NUST Bank "
                       "directly at +92 (51) 111 000 494.",
            )
    return GuardrailResult(is_safe=True)


def check_input_safety(text: str) -> GuardrailResult:
    """Run all input guardrails. Returns first failure or safe result."""
    # 1. Jailbreak detection
    result = detect_jailbreak(text)
    if not result.is_safe:
        return result

    # 2. Blocked topics
    result = detect_blocked_topics(text)
    if not result.is_safe:
        return result

    return GuardrailResult(is_safe=True)


# ── Output Guardrails (applied to LLM response) ─────────────────────────────

def scrub_pii_from_output(text: str) -> str:
    """Remove any PII that may have leaked into the response."""
    for label, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[REDACTED]", text)
    return text


def check_output_safety(text: str) -> GuardrailResult:
    """
    Verify the generated response is safe to show the user.
    Scrubs PII and checks for harmful content.
    """
    # Scrub PII
    cleaned = scrub_pii_from_output(text)

    # Check for disallowed content in output
    harmful_patterns = [
        r"(password|cvv|pin)\s*(is|:)\s*\S+",
        r"(account\s*number|card\s*number)\s*(is|:)\s*\d+",
    ]
    for pat in harmful_patterns:
        if re.search(pat, cleaned, re.IGNORECASE):
            logger.warning("Harmful content detected in output")
            return GuardrailResult(
                is_safe=False,
                reason="The response was filtered because it contained potentially "
                       "sensitive information. Please contact NUST Bank for assistance.",
                filtered_text="",
            )

    return GuardrailResult(is_safe=True, filtered_text=cleaned)


def is_out_of_domain(query: str, retrieved_scores: list) -> bool:
    """
    Determine if a query is out-of-domain based on retrieval scores.
    If all retrieved chunks have low similarity, the query is likely
    outside the bank's knowledge base.
    """
    if not retrieved_scores:
        return True
    avg_score = sum(retrieved_scores) / len(retrieved_scores)
    max_score = max(retrieved_scores)
    # Thresholds tuned for MiniLM cosine similarity
    return max_score < 0.25 and avg_score < 0.20
