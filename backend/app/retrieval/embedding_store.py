"""
backend/app/retrieval/embedding_store.py
---------------------------------
Sentence-transformer embedding model + ChromaDB persistent vector store
with cosine-similarity retrieval.
"""

import logging
from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer, models

from backend.app.core.settings import cfg

logger = logging.getLogger(__name__)


class EmbeddingStore:
    """
    Combines the embedding model and the ChromaDB vector index into one object.

    Responsibilities
    ----------------
    * Encode text with a sentence-transformer model.
    * Upsert encoded chunks into a ChromaDB persistent collection.
    * Answer similarity queries with cosine similarity via ChromaDB.
    """

    def __init__(self, persist_dir: str = cfg.paths.chroma_dir):
        logger.info("Loading embedding model: %s", cfg.embedding.model_name)
        # transformers 4.41+ hardcodes low_cpu_mem_usage=True which creates meta
        # tensors; sentence-transformers 2.x then calls .to(device) which fails.
        # Using models.Transformer with model_args lets us override this default.
        _transformer = models.Transformer(
            cfg.embedding.model_name,
            model_args={"low_cpu_mem_usage": False},
        )
        _pooling = models.Pooling(_transformer.get_word_embedding_dimension())
        self.model = SentenceTransformer(modules=[_transformer, _pooling])

        logger.info("Opening ChromaDB at: %s", persist_dir)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=cfg.embedding.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready — %d documents.",
            cfg.embedding.collection_name,
            self._collection.count(),
        )

    # ── Indexing ─────────────────────────────────────────────────────────────

    def index_documents(
        self,
        documents: List[Dict[str, str]],
        batch_size: int = cfg.embedding.batch_size,
    ) -> int:
        """
        Embed and upsert *documents* into the ChromaDB collection.

        Each dict must contain ``chunk_id``, ``content``, ``source``,
        and ``category`` keys.

        Returns the total number of documents processed.
        """
        if not documents:
            return 0

        total = 0
        for start in range(0, len(documents), batch_size):
            batch = documents[start: start + batch_size]
            ids = [doc["chunk_id"] for doc in batch]
            texts = [doc["content"] for doc in batch]
            metadatas = [
                {"source": doc["source"], "category": doc["category"]}
                for doc in batch
            ]
            embeddings = self.model.encode(texts, show_progress_bar=False).tolist()

            # upsert handles both new inserts and updates in one call
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            total += len(batch)
            logger.info("Upserted batch %d–%d into ChromaDB", start, start + len(batch))

        logger.info("Total documents in ChromaDB collection: %d", self._collection.count())
        return total

    # ── Retrieval ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = cfg.retriever.top_k) -> List[Dict]:
        """
        Return the *top_k* most relevant chunks for *query*.

        Each result dict contains:
            ``content``, ``source``, ``category``, ``score``

        ChromaDB cosine space returns distances in [0, 1] where 0 = identical.
        We convert to similarity: ``score = 1 - distance``.
        """
        if self._collection.count() == 0:
            return []

        query_vec = self.model.encode([query]).tolist()
        results = self._collection.query(
            query_embeddings=query_vec,
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append(
                {
                    "content": doc,
                    "source": meta.get("source", ""),
                    "category": meta.get("category", ""),
                    "score": round(1.0 - float(dist), 4),
                }
            )
        return output

    # ── Utilities ────────────────────────────────────────────────────────────

    def document_count(self) -> int:
        """Return the number of indexed document chunks."""
        return self._collection.count()

    def reset(self) -> None:
        """Wipe the entire ChromaDB collection and recreate it fresh."""
        self._client.delete_collection(cfg.embedding.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=cfg.embedding.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaDB collection reset.")
