"""
backend/app/ingestion/pipeline.py
--------------------------
Orchestrates the full document-ingestion pipeline:
  load → chunk → persist → return chunks
"""

import json
import logging
import os
from typing import List, Dict

from backend.app.core.settings import cfg
from backend.app.ingestion.excel_loader import ingest_excel
from backend.app.ingestion.json_loader import ingest_faq_json
from backend.app.ingestion.upload_loader import ingest_uploaded_documents
from backend.app.ingestion.chunker import chunk_documents

logger = logging.getLogger(__name__)


def load_all_documents() -> List[Dict[str, str]]:
    """
    Run the full ingestion pipeline and return chunked, ready-to-embed documents.

    Steps
    -----
    1. Load documents from Excel, FAQ JSON, and uploaded files.
    2. Chunk all documents with overlap.
    3. Persist the combined chunk list to ``data/processed/all_chunks.json``
       for reproducibility.
    """
    raw_docs: List[Dict[str, str]] = []
    raw_docs.extend(ingest_excel())
    raw_docs.extend(ingest_faq_json())
    raw_docs.extend(ingest_uploaded_documents())

    chunked = chunk_documents(raw_docs)

    # ── Persist for reproducibility ───────────────────────────────────────
    processed_path = os.path.join(cfg.paths.processed_dir, "all_chunks.json")
    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(chunked, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d chunks to %s", len(chunked), processed_path)

    return chunked
