# NUST Bank AI Support — Architecture Diagram

**CS416: Large Language Models | BESE-13 | LLM Implementation Submission**

---

## System Architecture Overview

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                     NUST Bank AI Customer Support System                   ║
║                    Retrieval-Augmented Generation (RAG)                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

 ┌────────────────────────────────────────────────────────────────────────┐
 │                         STREAMLIT WEB UI                               │
 │   ┌──────────────────────────┐   ┌────────────────────────────────┐   │
 │   │      Chat Interface      │   │           Sidebar              │   │
 │   │  • Message history       │   │  • System status / doc count   │   │
 │   │  • User input box        │   │  • File uploader (.txt/.json)  │   │
 │   │  • Source citations      │   │  • "Add to Knowledge Base" btn │   │
 │   │  • Response latency      │   │  • About / model info          │   │
 │   └──────────────┬───────────┘   └───────────────┬────────────────┘   │
 └──────────────────┼───────────────────────────────┼────────────────────┘
                    │ user query                     │ uploaded file
                    ▼                               ▼
 ┌──────────────────────────────┐   ┌───────────────────────────────────┐
 │      INPUT GUARDRAILS        │   │      INGESTION PIPELINE           │
 │  (src/core/guardrails.py)    │   │  (src/ingestion/pipeline.py)      │
 │                              │   │                                   │
 │  1. Jailbreak Detection      │   │  1. Excel Loader (openpyxl)       │
 │     • regex pattern match    │   │  2. JSON Loader (FAQ parser)      │
 │     • DAN / role-play inject │   │  3. Upload Loader (runtime files) │
 │                              │   │  4. Text Cleaner                  │
 │  2. Blocked Topic Filter     │   │     • normalise whitespace        │
 │     • PIN / CVV / password   │   │     • strip non-printable chars   │
 │     • exploit keywords       │   │  5. PII Anonymiser                │
 │                              │   │     • CNIC / phone / email        │
 │  → SAFE: continue pipeline   │   │     • account numbers            │
 │  → UNSAFE: return guardrail  │   │  6. Chunker                       │
 │    response immediately      │   │     • 500-char chunks             │
 └──────────────┬───────────────┘   │     • 50-char overlap            │
                │ safe query        └───────────────┬───────────────────┘
                ▼                                   │ chunk list
 ┌──────────────────────────────┐                   ▼
 │      EMBEDDING MODEL         │   ┌───────────────────────────────────┐
 │  sentence-transformers/      │   │        CHROMADB VECTOR STORE      │
 │  all-MiniLM-L6-v2  (22M)    │   │  (src/retrieval/embedding_store)  │
 │                              │   │                                   │
 │  • Encodes query to          │   │  • PersistentClient               │
 │    384-dim dense vector      │◄──┤  • Collection: nust_bank_knowledge│
 │                              │   │  • Distance metric: cosine        │
 │  • Same model used for       │   │  • HNSW index for fast ANN search │
 │    both indexing & search    │   │  • Upsert (insert + update)       │
 └──────────────┬───────────────┘   │  • Persisted to data/chroma_db/  │
                │ query vector      └───────────────────────────────────┘
                ▼
 ┌──────────────────────────────┐
 │      SEMANTIC RETRIEVAL      │
 │  ChromaDB .query()           │
 │                              │
 │  • Top-K=5 nearest chunks    │
 │  • cosine similarity scores  │
 │  • Returns: content, source, │
 │    category, score           │
 └──────────────┬───────────────┘
                │ retrieved chunks + scores
                ▼
 ┌──────────────────────────────┐
 │    OUT-OF-DOMAIN CHECK       │
 │  (src/core/guardrails.py)    │
 │                              │
 │  if max_score < 0.25 AND     │
 │     avg_score < 0.20:        │
 │       → OOD response         │
 │  else:                       │
 │       → continue             │
 └──────────────┬───────────────┘
                │ in-domain
                ▼
 ┌──────────────────────────────┐
 │      PROMPT ENGINEERING      │
 │  (src/core/prompt_engine.py) │
 │                              │
 │  Template:                   │
 │  ┌────────────────────────┐  │
 │  │ SYSTEM_PROMPT          │  │
 │  │ (7 behavioural rules)  │  │
 │  │                        │  │
 │  │ Context:               │  │
 │  │  [chunk_1]             │  │
 │  │  [chunk_2] ...         │  │
 │  │                        │  │
 │  │ Question: {query}      │  │
 │  │ Answer:                │  │
 │  └────────────────────────┘  │
 └──────────────┬───────────────┘
                │ formatted prompt
                ▼
 ┌──────────────────────────────┐
 │        LLM INFERENCE         │
 │  google/flan-t5-xl  (3B)     │
 │  (src/core/llm_engine.py)    │
 │                              │
 │  • AutoModelForSeq2SeqLM     │
 │  • max_new_tokens = 512      │
 │  • temperature = 0.3         │
 │  • top_p = 0.9               │
 │  • repetition_penalty = 1.2  │
 │  • do_sample = True          │
 │  • Device: CUDA / CPU        │
 └──────────────┬───────────────┘
                │ raw generated text
                ▼
 ┌──────────────────────────────┐
 │      OUTPUT GUARDRAILS       │
 │  (src/core/guardrails.py)    │
 │                              │
 │  1. PII Scrubber             │
 │     • regex replace:         │
 │       CNIC, phone, email,    │
 │       account numbers        │
 │     → [REDACTED]             │
 │                              │
 │  2. Harmful Content Filter   │
 │     • detect leaked secrets  │
 │     → return SAFETY_RESPONSE │
 │       if triggered           │
 └──────────────┬───────────────┘
                │ safe, clean response
                ▼
 ┌──────────────────────────────┐
 │      RESPONSE TO USER        │
 │                              │
 │  • Answer text               │
 │  • Source document labels    │
 │  • Response latency (ms)     │
 │  • Guardrail warning badge   │
 │    (if triggered)            │
 └──────────────────────────────┘
```

---

## Component Responsibilities

| Component | File | Role |
|---|---|---|
| **Settings** | `src/core/settings.py` | Single source of truth — all paths, model names, thresholds as frozen dataclasses |
| **Input Guardrails** | `src/core/guardrails.py` | Jailbreak + blocked-topic detection before any model call |
| **Output Guardrails** | `src/core/guardrails.py` | PII scrubbing + harmful content filter on generated text |
| **Prompt Engine** | `src/core/prompt_engine.py` | RAG prompt template, OOD response, safety response strings |
| **LLM Engine** | `src/core/llm_engine.py` | Orchestrates the full RAG pipeline end-to-end |
| **Text Cleaner** | `src/ingestion/text_cleaner.py` | Normalise whitespace, strip non-printables, anonymise PII |
| **Excel Loader** | `src/ingestion/excel_loader.py` | Parse product knowledge workbook with `openpyxl` |
| **JSON Loader** | `src/ingestion/json_loader.py` | Parse structured FAQ JSON |
| **Upload Loader** | `src/ingestion/upload_loader.py` | Load runtime user-uploaded `.txt`/.json files |
| **Chunker** | `src/ingestion/chunker.py` | Split documents into 500-char chunks with 50-char overlap |
| **Pipeline** | `src/ingestion/pipeline.py` | Orchestrates all loaders → chunker → persist to `all_chunks.json` |
| **Embedding Store** | `src/retrieval/embedding_store.py` | `all-MiniLM-L6-v2` encoder + ChromaDB persistent collection |
| **Styles** | `src/ui/styles.py` | Custom CSS injection for Streamlit |
| **Sidebar** | `src/ui/sidebar.py` | System stats, document upload panel |
| **Chat** | `src/ui/chat.py` | Chat history, user input, response rendering with sources |
| **App Entry** | `app.py` | Thin Streamlit entry point — wires all components together |

---

## Data Flow Summary

```
assets/                       ──► Ingestion Pipeline ──► ChromaDB (data/chroma_db/)
  NUST Bank-Product-Knowledge.xlsx                            ▲
  funds_transfer_app_features_faq.json                       │
                                                             │ upsert
data/uploaded_docs/           ──► Upload Loader   ──────────┘
  (runtime user uploads)

User Query ──► Input Guardrails ──► Embedding ──► ChromaDB Query
           ──► OOD Check ──► Prompt Build ──► Flan-T5-XL ──► Output Guardrails
           ──► Streamlit UI (answer + sources + latency)
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **RAG over pure fine-tuning** | Allows real-time knowledge updates without retraining; grounding reduces hallucinations |
| **ChromaDB over NumPy** | Persistent HNSW index scales to large document sets; native cosine distance; production-grade upsert |
| **Flan-T5-XL (3B)** | Best seq2seq model under the 6B limit; no gating; Apache 2.0; instruction-tuned |
| **LoRA for fine-tuning** | Trains only 0.1% of parameters; feasible on CPU/single GPU; adapter weights are small and portable |
| **Frozen dataclass config** | Prevents accidental mutation; all settings in one place; easy to extend |
| **Modular `src/` package** | Each concern is isolated; testable independently; no circular dependencies |
