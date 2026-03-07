"""
src/ingestion/text_cleaner.py
------------------------------
Stateless utilities for cleaning raw text and anonymising PII.
"""

import re
from src.core.settings import cfg


def clean_text(text: str) -> str:
    """
    Normalise raw text:
    - Replace non-breaking spaces / tabs
    - Strip non-printable characters
    - Collapse repeated whitespace and blank lines
    """
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\t", " ")
    text = re.sub(r"[^\x20-\x7E\n\r]", " ", text)
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def anonymize_text(text: str) -> str:
    """Replace all PII matches with a labelled placeholder."""
    for label, pattern in cfg.guardrails.pii_patterns.items():
        text = re.sub(pattern, f"[REDACTED_{label.upper()}]", text)
    return text
