"""
src/ingestion/json_loader.py
-----------------------------
Ingests FAQ data from the structured JSON knowledge-base file.
Each Q&A pair becomes a separate document tagged with its category.
"""

import json
import logging
from typing import List, Dict

from src.core.settings import cfg
from src.ingestion.text_cleaner import clean_text, anonymize_text

logger = logging.getLogger(__name__)


def ingest_faq_json(path: str = cfg.paths.faq_json_path) -> List[Dict[str, str]]:
    """
    Read the FAQ JSON file.

    Expected schema::

        {
          "categories": [
            {
              "category": "...",
              "questions": [{"question": "...", "answer": "..."}, ...]
            },
            ...
          ]
        }

    Returns a list of document dicts:
        {"source": str, "category": str, "content": str}
    """
    documents: List[Dict[str, str]] = []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

        for cat in data.get("categories", []):
            category = cat.get("category", "General")
            for i, qa in enumerate(cat.get("questions", [])):
                question = clean_text(qa.get("question", ""))
                answer = clean_text(qa.get("answer", ""))
                if question or answer:
                    content = anonymize_text(f"Question: {question}\nAnswer: {answer}")
                    documents.append({
                        "source": f"FAQ_JSON/{category}/Q_{i}",
                        "category": category,
                        "content": content,
                    })

    logger.info("Ingested %d FAQ items from JSON", len(documents))
    return documents
