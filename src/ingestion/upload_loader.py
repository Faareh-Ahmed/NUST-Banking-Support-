"""
src/ingestion/upload_loader.py
-------------------------------
Ingests user-uploaded .txt and .json files from the uploads directory.
Supports both the FAQ JSON schema and arbitrary plain-text/JSON files.
"""

import json
import logging
import os
from typing import List, Dict

from src.core.settings import cfg
from src.ingestion.text_cleaner import clean_text, anonymize_text

logger = logging.getLogger(__name__)


def ingest_uploaded_documents(directory: str = cfg.paths.uploaded_docs_dir) -> List[Dict[str, str]]:
    """
    Read all ``.txt`` and ``.json`` files in *directory*.

    - ``.txt`` files → cleaned and anonymised plain text.
    - ``.json`` files that follow the FAQ schema → delegated to
      :func:`~src.ingestion.json_loader.ingest_faq_json`.
    - Other ``.json`` files → serialised as text.

    Returns a list of document dicts:
        {"source": str, "category": str, "content": str}
    """
    # Deferred import to avoid circular dependency
    from src.ingestion.json_loader import ingest_faq_json

    documents: List[Dict[str, str]] = []
    if not os.path.isdir(directory):
        logger.warning("Upload directory does not exist: %s", directory)
        return documents

    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)

        if fname.endswith(".txt"):
            with open(fpath, "r", encoding="utf-8") as f:
                content = anonymize_text(clean_text(f.read()))
            documents.append({
                "source": f"uploaded/{fname}",
                "category": "Uploaded",
                "content": content,
            })

        elif fname.endswith(".json"):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "categories" in data:
                    # Reuse the FAQ ingestion logic
                    documents.extend(ingest_faq_json(fpath))
                else:
                    content = anonymize_text(clean_text(json.dumps(data, indent=2)))
                    documents.append({
                        "source": f"uploaded/{fname}",
                        "category": "Uploaded",
                        "content": content,
                    })
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON file: %s", fname)

    logger.info("Ingested %d uploaded documents", len(documents))
    return documents
