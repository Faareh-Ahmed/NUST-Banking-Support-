"""
backend/app/core/llm_engine.py
-----------------------
End-to-end Retrieval-Augmented Generation (RAG) pipeline.

LLM inference is delegated to the Groq API (cloud-hosted Llama 3.2 3B).
The embedding model and vector store still run locally.

Flow
----
User Query
  → Input Guardrails
  → Embedding + ChromaDB Retrieval (local)
  → Out-of-Domain Check
  → Groq API: Llama 3.2 3B inference
  → Output Guardrails
  → Response
"""

import logging
import os
import time
from typing import Dict, List, Optional

from groq import Groq

from backend.app.core.settings import cfg
from backend.app.core.guardrails import (
    check_input_safety,
    check_output_safety,
    is_out_of_domain,
)
from backend.app.core.prompt_engine import (
    OUT_OF_DOMAIN_RESPONSE,
    SAFETY_RESPONSE,
    SYSTEM_PROMPT,
)
from backend.app.retrieval.embedding_store import EmbeddingStore

logger = logging.getLogger(__name__)


class LLMEngine:
    """
    Orchestrates the full RAG pipeline for a single user query.

    Parameters
    ----------
    embedding_store:
        An already-initialised EmbeddingStore.
        If None, a fresh one is created from the persisted ChromaDB index.
    """

    def __init__(self, embedding_store: Optional[EmbeddingStore] = None):
        self.store = embedding_store or EmbeddingStore()

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at https://console.groq.com and add it to backend/.env"
            )
        self._client = Groq(api_key=api_key)
        logger.info("Groq API client ready — model: %s", cfg.llm.model_name)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate(self, query: str, context_chunks: List[Dict]) -> str:
        """Send a chat-completion request to the Groq API and return the answer text."""
        context_text = "\n\n".join(chunk["content"] for chunk in context_chunks)

        completion = self._client.chat.completions.create(
            model=cfg.llm.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Answer the question using ONLY the context provided below. "
                        "Be specific, accurate, and helpful. "
                        "Use bullet points when listing multiple items.\n\n"
                        f"Context:\n{context_text}\n\n"
                        f"Question: {query}"
                    ),
                },
            ],
            temperature=cfg.llm.temperature,
            max_tokens=cfg.llm.max_new_tokens,
        )
        return completion.choices[0].message.content or ""

    @staticmethod
    def _make_result(
        answer: str = "",
        sources: Optional[List[str]] = None,
        latency_ms: float = 0.0,
        guardrail_triggered: bool = False,
        out_of_domain: bool = False,
    ) -> Dict:
        return {
            "answer": answer,
            "sources": sources or [],
            "latency_ms": latency_ms,
            "guardrail_triggered": guardrail_triggered,
            "out_of_domain": out_of_domain,
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def answer(self, query: str) -> Dict:
        """
        Process *query* through the full RAG pipeline.

        Returns
        -------
        dict
            Keys: answer, sources, latency_ms, guardrail_triggered, out_of_domain
        """
        start = time.time()

        def elapsed() -> float:
            return round((time.time() - start) * 1000, 1)

        # Step 1 — Input guardrails (local, instant)
        input_check = check_input_safety(query)
        if not input_check.is_safe:
            return self._make_result(
                answer=input_check.reason,
                guardrail_triggered=True,
                latency_ms=elapsed(),
            )

        # Step 2 — Semantic retrieval from ChromaDB (local)
        retrieved = self.store.search(query, top_k=cfg.retriever.top_k)

        # Step 3 — Out-of-domain check (cosine similarity threshold)
        scores = [r["score"] for r in retrieved]
        if is_out_of_domain(query, scores):
            return self._make_result(
                answer=OUT_OF_DOMAIN_RESPONSE,
                out_of_domain=True,
                latency_ms=elapsed(),
            )

        # Step 4 — Generate answer via Groq API
        raw_answer = self._generate(query, retrieved).strip()

        # Step 5 — Output guardrails (PII scrub + harmful content check)
        output_check = check_output_safety(raw_answer)
        if not output_check.is_safe:
            return self._make_result(
                answer=SAFETY_RESPONSE,
                sources=list({r["source"] for r in retrieved}),
                guardrail_triggered=True,
                latency_ms=elapsed(),
            )

        return self._make_result(
            answer=output_check.filtered_text,
            sources=list({r["source"] for r in retrieved}),
            latency_ms=elapsed(),
        )
