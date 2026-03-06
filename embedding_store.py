"""
Embedding & Vector Store Module
--------------------------------
Creates sentence-transformer embeddings for document chunks and
stores them in a persistent JSON-based vector store with numpy
cosine similarity for fast retrieval.
"""

import json
import logging
import os
from typing import List, Dict, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_DIR,
    COLLECTION_NAME,
    TOP_K_RESULTS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Persistent storage path
VECTOR_STORE_PATH = os.path.join(CHROMA_DIR, f"{COLLECTION_NAME}.json")


class EmbeddingStore:
    """Wraps embedding model + persistent vector index for indexing & retrieval."""

    def __init__(self):
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        # In-memory store
        self._ids: List[str] = []
        self._embeddings: List[List[float]] = []
        self._documents: List[str] = []
        self._metadatas: List[Dict] = []

        # Load persisted data if available
        self._load()
        logger.info(
            "Vector store '%s' has %d documents",
            COLLECTION_NAME,
            len(self._ids),
        )

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self):
        """Load the vector store from disk."""
        if os.path.exists(VECTOR_STORE_PATH):
            try:
                with open(VECTOR_STORE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._ids = data.get("ids", [])
                self._embeddings = data.get("embeddings", [])
                self._documents = data.get("documents", [])
                self._metadatas = data.get("metadatas", [])
                logger.info("Loaded %d documents from disk", len(self._ids))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load vector store: %s — starting fresh", e)

    def _save(self):
        """Persist the vector store to disk."""
        os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
        data = {
            "ids": self._ids,
            "embeddings": self._embeddings,
            "documents": self._documents,
            "metadatas": self._metadatas,
        }
        with open(VECTOR_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info("Saved %d documents to disk", len(self._ids))

    # ── Indexing ─────────────────────────────────────────────────────────────

    def index_documents(self, documents: List[Dict[str, str]], batch_size: int = 64) -> int:
        """
        Embed and upsert a list of document chunks into the vector store.
        Each dict must have 'chunk_id', 'content', 'source', 'category'.
        Returns the number of documents indexed.
        """
        if not documents:
            return 0

        total = 0
        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            ids = [doc["chunk_id"] for doc in batch]
            texts = [doc["content"] for doc in batch]
            metadatas = [
                {"source": doc["source"], "category": doc["category"]}
                for doc in batch
            ]

            embeddings = self.model.encode(texts, show_progress_bar=False).tolist()

            # Upsert: update existing or add new
            for i, doc_id in enumerate(ids):
                if doc_id in self._ids:
                    idx = self._ids.index(doc_id)
                    self._embeddings[idx] = embeddings[i]
                    self._documents[idx] = texts[i]
                    self._metadatas[idx] = metadatas[i]
                else:
                    self._ids.append(doc_id)
                    self._embeddings.append(embeddings[i])
                    self._documents.append(texts[i])
                    self._metadatas.append(metadatas[i])

            total += len(batch)
            logger.info("Indexed batch %d-%d", start, start + len(batch))

        self._save()
        logger.info("Total documents in store: %d", len(self._ids))
        return total

    # ── Retrieval ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """
        Retrieve the top-k most relevant chunks for a query using
        cosine similarity.
        Returns list of dicts with keys: content, source, category, score.
        """
        if not self._ids:
            return []

        query_embedding = self.model.encode([query])[0]

        # Compute cosine similarities
        store_embeddings = np.array(self._embeddings)
        query_vec = np.array(query_embedding)

        dot_products = store_embeddings @ query_vec
        query_norm = np.linalg.norm(query_vec)
        store_norms = np.linalg.norm(store_embeddings, axis=1)
        similarities = dot_products / (store_norms * query_norm + 1e-10)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        retrieved: List[Dict] = []
        for idx in top_indices:
            retrieved.append(
                {
                    "content": self._documents[idx],
                    "source": self._metadatas[idx].get("source", ""),
                    "category": self._metadatas[idx].get("category", ""),
                    "score": round(float(similarities[idx]), 4),
                }
            )
        return retrieved

    # ── Utilities ────────────────────────────────────────────────────────────

    def reset_collection(self):
        """Delete all data and reset the store."""
        self._ids.clear()
        self._embeddings.clear()
        self._documents.clear()
        self._metadatas.clear()
        if os.path.exists(VECTOR_STORE_PATH):
            os.remove(VECTOR_STORE_PATH)
        logger.info("Vector store reset.")

    def document_count(self) -> int:
        return len(self._ids)


if __name__ == "__main__":
    from data_ingestion import load_all_documents

    store = EmbeddingStore()
    if store.document_count() == 0:
        docs = load_all_documents()
        store.index_documents(docs)
    print(f"\nIndexed {store.document_count()} chunks.")

    # Quick test
    test_queries = [
        "What is the transfer limit?",
        "Who can apply for auto finance?",
        "What types of accounts does NUST Bank offer?",
    ]
    for q in test_queries:
        print(f"\n{'='*50}")
        print(f"Q: {q}")
        results = store.search(q)
        for r in results[:3]:
            print(f"  [{r['score']:.3f}] {r['source']}")
            print(f"  {r['content'][:150]}...")
