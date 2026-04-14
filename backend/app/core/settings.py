"""
backend/app/core/settings.py
---------------------
Centralised, typed configuration for the NUST Bank AI Support system.
All settings are grouped into frozen dataclasses so they can be imported
individually or as a whole via the top-level `cfg` singleton.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

# ── Base directory resolution ─────────────────────────────────────────────────
# Points to the project root (four levels above this file:
# backend/app/core/settings.py → backend/app/core → backend/app → backend → project root)
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])


# ── Path Settings ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PathSettings:
    base_dir: str = _PROJECT_ROOT
    assets_dir: str = os.path.join(_PROJECT_ROOT, "assets")
    data_dir: str = os.path.join(_PROJECT_ROOT, "data")
    chroma_dir: str = os.path.join(_PROJECT_ROOT, "data", "chroma_db")
    processed_dir: str = os.path.join(_PROJECT_ROOT, "data", "processed")
    uploaded_docs_dir: str = os.path.join(_PROJECT_ROOT, "data", "uploaded_docs")
    # Source knowledge files live under assets/
    excel_path: str = os.path.join(_PROJECT_ROOT, "assets", "NUST Bank-Product-Knowledge.xlsx")
    faq_json_path: str = os.path.join(_PROJECT_ROOT, "assets", "funds_transfer_app_features_faq.json")

    def ensure_dirs(self) -> None:
        """Create all required directories if they do not exist."""
        for d in [self.assets_dir, self.data_dir, self.chroma_dir,
                  self.processed_dir, self.uploaded_docs_dir]:
            os.makedirs(d, exist_ok=True)


# ── Embedding Model Settings ──────────────────────────────────────────────────

@dataclass(frozen=True)
class EmbeddingSettings:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    collection_name: str = "nust_bank_knowledge"
    batch_size: int = 64


# ── LLM Settings ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LLMSettings:
    # Groq-hosted Llama 3.2 — 3B parameters, free tier, ~200ms responses.
    # Alternative Groq models (all free, all under 6B):
    #   "llama-3.2-1b-preview"    # 1B  — fastest, lower quality
    #   "llama-3.2-3b-preview"    # 3B  — recommended sweet spot
    # Inference runs on Groq's LPU hardware; no local GPU needed.
    model_name: str = "llama-3.2-3b-preview"
    max_new_tokens: int = 400
    temperature: float = 0.3


# ── Retriever Settings ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RetrieverSettings:
    chunk_size: int = 500       # characters per chunk
    chunk_overlap: int = 50     # characters of overlap between chunks
    top_k: int = 3              # 3 chunks keeps prompt within 512-token T5 limit
    # Out-of-domain thresholds (cosine similarity)
    ood_max_score_threshold: float = 0.25
    ood_avg_score_threshold: float = 0.20


# ── Guardrail Settings ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GuardrailSettings:
    pii_patterns: Dict[str, str] = field(default_factory=lambda: {
        "cnic":           r"\b\d{5}-\d{7}-\d{1}\b",
        "phone_pk":       r"\b(?:\+92|0)\s*\d{3}[\s-]?\d{7}\b",
        "email":          r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "account_number": r"\b\d{10,16}\b",
        "credit_card":    r"\b(?:\d{4}[\s-]?){3}\d{4}\b",
    })

    blocked_topics: List[str] = field(default_factory=lambda: [
        "password", "pin number", "cvv", "secret question",
        "social security", "hack", "exploit", "bypass security",
        "ignore previous instructions", "ignore above",
        "disregard your instructions", "pretend you are",
        "act as if you have no restrictions",
        "reveal system prompt", "show me your prompt",
        "what are your instructions", "repeat your system message",
    ])

    jailbreak_patterns: List[str] = field(default_factory=lambda: [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
        r"(pretend|act|behave)\s+(as\s+if|like)\s+you\s+(are|have|can)",
        r"(disregard|forget|override)\s+(your|all|the)\s+(rules|guidelines|restrictions|instructions)",
        r"you\s+are\s+now\s+(DAN|unrestricted|unfiltered|jailbroken)",
        r"(reveal|show|tell\s+me|display)\s+(your|the)\s+(system\s+)?(prompt|instructions|rules)",
        r"(do\s+not|don'?t)\s+(follow|obey|listen\s+to)\s+(your|the)\s+(rules|guidelines)",
    ])

    harmful_output_patterns: List[str] = field(default_factory=lambda: [
        r"(password|cvv|pin)\s*(is|:)\s*\S+",
        r"(account\s*number|card\s*number)\s*(is|:)\s*\d+",
    ])


# ── Root Config Singleton ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class AppConfig:
    paths: PathSettings = field(default_factory=PathSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    retriever: RetrieverSettings = field(default_factory=RetrieverSettings)
    guardrails: GuardrailSettings = field(default_factory=GuardrailSettings)


# Module-level singleton — import this everywhere:  from backend.app.core.settings import cfg
cfg = AppConfig()

# Ensure required directories exist on import
cfg.paths.ensure_dirs()
