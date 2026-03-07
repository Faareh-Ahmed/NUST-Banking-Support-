"""
src/ingestion/chunker.py
-------------------------
Splits documents into overlapping text chunks suitable for embedding.
"""

import logging
from typing import List, Dict

from src.core.settings import cfg

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = cfg.retriever.chunk_size,
    overlap: int = cfg.retriever.chunk_overlap,
) -> List[str]:
    """
    Split *text* into overlapping chunks of roughly *chunk_size* characters.

    The splitter tries to break on sentence boundaries (period or newline)
    within the latter half of each window, falling back to a hard cut when
    no such boundary is found.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Prefer breaking at the last sentence boundary in the second half
        last_period = chunk.rfind(".")
        last_newline = chunk.rfind("\n")
        break_pos = max(last_period, last_newline)

        if break_pos > chunk_size // 2:
            end = start + break_pos + 1
            chunk = text[start:end]

        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if c]


def chunk_documents(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Chunk each document and propagate its metadata to every child chunk.

    Each output dict contains:
        ``source``, ``category``, ``content``, ``chunk_id``
    """
    chunked: List[Dict[str, str]] = []
    for doc in documents:
        for i, chunk in enumerate(chunk_text(doc["content"])):
            chunked.append({
                "source": doc["source"],
                "category": doc["category"],
                "content": chunk,
                "chunk_id": f"{doc['source']}__chunk_{i}",
            })

    logger.info("Created %d chunks from %d documents", len(chunked), len(documents))
    return chunked
