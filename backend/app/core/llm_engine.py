"""
backend/app/core/llm_engine.py
-----------------------
End-to-end Retrieval-Augmented Generation (RAG) pipeline.

LLM: Qwen/Qwen3-1.7B running locally on CPU via the transformers library.
Model is downloaded once (~3.4 GB) and cached in ~/.cache/huggingface/.
No API key or GPU required.

Flow
----
User Query
  → Input Guardrails          (regex, <1 ms)
  → Embedding + ChromaDB      (all-MiniLM-L6-v2, local)
  → Out-of-Domain Check       (cosine similarity thresholds)
  → Qwen3-1.7B local generate (CPU, bfloat16, ~30–90 s first run)
  → Output Guardrails         (PII scrub + harmful-content check)
  → Response
"""

import logging
import time
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

        logger.info("Loading tokenizer: %s", cfg.llm.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.llm.model_name,
            trust_remote_code=True,
        )

        # bfloat16 halves RAM usage (~3.4 GB vs ~6.8 GB) and is natively
        # supported by Intel Core Ultra processors.
        logger.info(
            "Loading model %s on CPU (bfloat16) — first run downloads ~3.4 GB...",
            cfg.llm.model_name,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.llm.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,    # load weights shard-by-shard to limit peak RAM
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("Model loaded on CPU — ready.")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _generate(self, query: str, context_chunks: List[Dict]) -> str:
        """Format messages, run one forward pass, return the generated text."""
        context_text = "\n\n".join(chunk["content"] for chunk in context_chunks)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Answer using ONLY the context below. "
                    "Be specific and helpful. Use bullet points for lists.\n\n"
                    f"Context:\n{context_text}\n\n"
                    f"Question: {query}"
                ),
            },
        ]

        # Apply the Qwen3 chat template.
        # enable_thinking=False disables the <think>...</think> reasoning step
        # so we get the answer directly without extra latency.
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older tokenizer versions that don't support enable_thinking
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=cfg.llm.max_new_tokens,
                temperature=cfg.llm.temperature,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens — not the prompt
        new_ids = output_ids[0][prompt_len:]
        answer = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # Strip any residual <think>...</think> block the model may have emitted
        if "<think>" in answer:
            after_think = answer.split("</think>", 1)
            answer = after_think[-1].strip() if len(after_think) > 1 else answer

        return answer

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

        # Step 3 — Out-of-domain check
        scores = [r["score"] for r in retrieved]
        if is_out_of_domain(query, scores):
            return self._make_result(
                answer=OUT_OF_DOMAIN_RESPONSE,
                out_of_domain=True,
                latency_ms=elapsed(),
            )

        # Step 4 — Generate answer locally
        raw_answer = self._generate(query, retrieved)

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
