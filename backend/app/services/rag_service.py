"""Application service that bridges FastAPI endpoints with RAG engine."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass

from backend.app.core.llm_engine import LLMEngine
from backend.app.core.settings import cfg
from backend.app.ingestion.chunker import chunk_documents
from backend.app.ingestion.upload_loader import ingest_uploaded_documents
from backend.app.retrieval.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)


@dataclass
class RAGService:
    """Lazy-load and expose operations needed by API handlers."""

    _store: EmbeddingStore | None = None
    _engine: LLMEngine | None = None
    _store_error: str | None = None

    # Session-level metrics (reset on server restart)
    _total_queries: int = 0
    _total_latency_ms: float = 0.0
    _guardrail_triggers: int = 0
    _out_of_domain_count: int = 0

    def _ensure_ready_store(self) -> None:
        """Initialize vector store once per process."""
        if self._store is None:
            if self._store_error is not None:
                # ChromaDB already failed; don't retry
                logger.warning("ChromaDB already failed in this process; skipping retry.")
                raise RuntimeError(self._store_error)

            try:
                self._store = EmbeddingStore()
                if self._store.document_count() == 0:
                    from backend.app.ingestion.pipeline import load_all_documents

                    logger.info("No embeddings found. Running initial ingestion/indexing.")
                    docs = load_all_documents()
                    self._store.index_documents(docs)
            except Exception as e:
                logger.error(f"ChromaDB initialization failed: {e}")
                self._store_error = str(e)
                self._store = None
                # Attempt to reset the corrupted database
                if os.path.exists(cfg.paths.chroma_dir):
                    logger.info(f"Resetting corrupted ChromaDB at {cfg.paths.chroma_dir}")
                    try:
                        shutil.rmtree(cfg.paths.chroma_dir)
                        os.makedirs(cfg.paths.chroma_dir, exist_ok=True)
                    except Exception as reset_error:
                        logger.error(f"Failed to reset ChromaDB: {reset_error}")
                raise

    def _ensure_ready_engine(self) -> None:
        """Initialize LLM once per process."""
        self._ensure_ready_store()
        if self._engine is None:
            self._engine = LLMEngine(embedding_store=self._store)

    def health(self) -> dict:
        return {"status": "ok", "service": "nust-bank-backend"}

    def stats(self) -> dict:
        avg = round(self._total_latency_ms / self._total_queries, 1) if self._total_queries else 0.0
        try:
            self._ensure_ready_store()
            assert self._store is not None
            return {
                "indexed_documents": self._store.document_count(),
                "llm_model": cfg.llm.model_name,
                "embedding_model": cfg.embedding.model_name,
                "total_queries": self._total_queries,
                "avg_latency_ms": avg,
                "guardrail_triggers": self._guardrail_triggers,
                "out_of_domain_count": self._out_of_domain_count,
            }
        except Exception as e:
            logger.error(f"Stats endpoint failed: {e}")
            return {
                "indexed_documents": 0,
                "llm_model": cfg.llm.model_name,
                "embedding_model": cfg.embedding.model_name,
                "total_queries": self._total_queries,
                "avg_latency_ms": avg,
                "guardrail_triggers": self._guardrail_triggers,
                "out_of_domain_count": self._out_of_domain_count,
            }

    def chat(self, message: str) -> dict:
        self._ensure_ready_engine()
        assert self._engine is not None
        result = self._engine.answer(message)
        self._total_queries += 1
        self._total_latency_ms += result.get("latency_ms", 0.0)
        if result.get("guardrail_triggered"):
            self._guardrail_triggers += 1
        if result.get("out_of_domain"):
            self._out_of_domain_count += 1
        return result

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