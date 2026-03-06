"""
Configuration settings for NUST Bank LLM Customer Support System.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

EXCEL_PATH = os.path.join(BASE_DIR, "NUST Bank-Product-Knowledge.xlsx")
FAQ_JSON_PATH = os.path.join(BASE_DIR, "funds_transfer_app_features_faq.json")
UPLOADED_DOCS_DIR = os.path.join(DATA_DIR, "uploaded_docs")

# Create directories if they don't exist
for d in [DATA_DIR, CHROMA_DIR, PROCESSED_DIR, UPLOADED_DOCS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# ── LLM Settings ──────────────────────────────────────────────────────────────
# Using Flan-T5-base (250M params) — well within 6B limit, fast inference
# Alternatives: google/flan-t5-large (780M), google/flan-t5-xl (3B)
LLM_MODEL_NAME = "google/flan-t5-base"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.3
TOP_P = 0.9
REPETITION_PENALTY = 1.2

# ── Retrieval Settings ────────────────────────────────────────────────────────
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 50        # overlap between chunks
TOP_K_RESULTS = 5         # number of retrieved chunks for context

# ── ChromaDB Collection ──────────────────────────────────────────────────────
COLLECTION_NAME = "nust_bank_knowledge"

# ── PII Patterns for Anonymization ───────────────────────────────────────────
PII_PATTERNS = {
    "cnic":           r"\b\d{5}-\d{7}-\d{1}\b",
    "phone_pk":       r"\b(?:\+92|0)\s*\d{3}[\s-]?\d{7}\b",
    "email":          r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "account_number": r"\b\d{10,16}\b",
    "credit_card":    r"\b(?:\d{4}[\s-]?){3}\d{4}\b",
}

# ── Guardrail Keywords ───────────────────────────────────────────────────────
BLOCKED_TOPICS = [
    "password", "pin number", "cvv", "secret question",
    "social security", "hack", "exploit", "bypass security",
    "ignore previous instructions", "ignore above",
    "disregard your instructions", "pretend you are",
    "act as if you have no restrictions",
    "reveal system prompt", "show me your prompt",
    "what are your instructions", "repeat your system message",
]

JAILBREAK_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"(pretend|act|behave)\s+(as\s+if|like)\s+you\s+(are|have|can)",
    r"(disregard|forget|override)\s+(your|all|the)\s+(rules|guidelines|restrictions|instructions)",
    r"you\s+are\s+now\s+(DAN|unrestricted|unfiltered|jailbroken)",
    r"(reveal|show|tell\s+me|display)\s+(your|the)\s+(system\s+)?(prompt|instructions|rules)",
    r"(do\s+not|don'?t)\s+(follow|obey|listen\s+to)\s+(your|the)\s+(rules|guidelines)",
]
