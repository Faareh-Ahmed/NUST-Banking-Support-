"""Application service that bridges FastAPI endpoints with RAG engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.core.llm_engine import LLMEngine
from src.core.settings import cfg
from src.ingestion.chunker import chunk_documents
from src.ingestion.upload_loader import ingest_uploaded_documents
from src.retrieval.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)


@dataclass
class RAGService:
    """Lazy-load and expose operations needed by API handlers."""

    _store: EmbeddingStore | None = None
    _engine: LLMEngine | None = None

    def _ensure_ready_store(self) -> None:
        """Initialize vector store once per process."""
        if self._store is None:
            self._store = EmbeddingStore()
            if self._store.document_count() == 0:
                from src.ingestion.pipeline import load_all_documents

                logger.info("No embeddings found. Running initial ingestion/indexing.")
                docs = load_all_documents()
                self._store.index_documents(docs)

    def _ensure_ready_engine(self) -> None:
        """Initialize LLM once per process."""
        self._ensure_ready_store()
        if self._engine is None:
            self._engine = LLMEngine(embedding_store=self._store)

    def health(self) -> dict:
        return {"status": "ok", "service": "nust-bank-backend"}

    def stats(self) -> dict:
        self._ensure_ready_store()
        assert self._store is not None

        return {
            "indexed_documents": self._store.document_count(),
            "llm_model": cfg.llm.model_name,
            "embedding_model": cfg.embedding.model_name,
        }

    def chat(self, message: str) -> dict:
        self._ensure_ready_engine()
        assert self._engine is not None
        return self._engine.answer(message)

    def upload_and_index(self, filename: str) -> dict:
        """
        Re-ingest uploaded docs directory and upsert chunks.

        Upsert IDs are deterministic, so existing uploaded files update in-place.
        """
        self._ensure_ready_store()
        assert self._store is not None

        docs = ingest_uploaded_documents(cfg.paths.uploaded_docs_dir)
        chunks = chunk_documents(docs)
        indexed = self._store.index_documents(chunks) if chunks else 0

        return {
            "filename": filename,
            "indexed_chunks": indexed,
            "indexed_documents_total": self._store.document_count(),
        }