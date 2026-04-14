# NUST Bank AI Customer Support System

An end-to-end **Retrieval-Augmented Generation (RAG)** chatbot for NUST Bank customer support. The system answers banking queries by retrieving relevant passages from the bank's own knowledge base and generating grounded, verified responses via a cloud-hosted Large Language Model. A Next.js web interface and a FastAPI backend serve the complete experience.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Component Breakdown](#4-component-breakdown)
5. [Chat Flow — Step by Step](#5-chat-flow--step-by-step)
6. [Document Upload & Ingestion Flow](#6-document-upload--ingestion-flow)
7. [Chunking Algorithm — In Detail](#7-chunking-algorithm--in-detail)
8. [Embedding & Vector Store](#8-embedding--vector-store)
9. [Retrieval Mechanism](#9-retrieval-mechanism)
10. [Guardrail System](#10-guardrail-system)
11. [LLM — Groq API (Llama 3.2 3B)](#11-llm--groq-api-llama-32-3b)
12. [API Reference](#12-api-reference)
13. [Project Structure](#13-project-structure)
14. [Configuration Reference](#14-configuration-reference)
15. [Setup & Running](#15-setup--running)

---

## 1. System Overview

The NUST Bank AI Customer Support System solves a fundamental problem in deploying LLMs for enterprise use: **hallucination**. Generic LLMs have no knowledge of NUST Bank's specific products, policies, and fee structures. Instead of fine-tuning (expensive, slow to update) or prompting with the entire knowledge base (context-window limited), this system uses **Retrieval-Augmented Generation (RAG)**:

1. The bank's knowledge base is pre-processed, split into small passages, and stored as vector embeddings in a database.
2. When a user asks a question, the system finds the most semantically relevant passages from that database.
3. Only those passages (not the entire knowledge base) are sent to the LLM along with the question.
4. The LLM generates an answer **grounded in the retrieved passages** — it cannot fabricate facts about products it was not given.

This approach delivers **accurate, up-to-date, source-grounded** answers while keeping API costs minimal (3 short passages per query, not thousands of documents).

---

## 2. High-Level Architecture

```
+--------------------------------------------------------------------------+
|                         USER'S BROWSER                                   |
|  +-------------------------------------------------------------------+   |
|  |              Next.js 14 Frontend  (port 3000)                     |   |
|  |  - Chat UI (messages, latency badge, guardrail warnings)          |   |
|  |  - Upload modal (drag-drop .txt / .json)                          |   |
|  |  - System status bar (indexed chunks, model names)                |   |
|  +------------------+----------------------+---------------------------+   |
+---------------------|-----------------------|---------------------------+
                       |  POST /chat           |  POST /upload
                       |  GET  /stats          |  GET  /health
                       v                       v
+--------------------------------------------------------------------------+
|                    FastAPI Backend  (port 8000)                          |
|                                                                          |
|  +-------------------------------------------------------------------+  |
|  |                       RAGService                                   |  |
|  |  (lazy-init singleton: initialises store & engine on first use)    |  |
|  |                           |                                        |  |
|  |          +----------------+----------------+                       |  |
|  |          v                                 v                       |  |
|  |   +--------------+                 +------------------+            |  |
|  |   | EmbeddingStore|                |   LLMEngine       |            |  |
|  |   | (local)       |                |                  |            |  |
|  |   |               |                | 1. Input Guards  |            |  |
|  |   | SentenceTrans |                | 2. -> EmbedStore |            |  |
|  |   | + ChromaDB    |<---------------| 3. OOD Check     |            |  |
|  |   +---------------+                | 4. Groq API call |------> GROQ CLOUD
|  |                                    | 5. Output Guards |        Llama 3.2 3B
|  +------------------------------------+------------------+            |  |
|                                                                          |
|  +-------------------------------------------------------------------+  |
|  |                    Ingestion Pipeline                              |  |
|  |  Excel Loader -> JSON Loader -> Upload Loader                      |  |
|  |          +----------------> TextCleaner + Chunker                  |  |
|  +------------------------------+------------------------------------+  |
+-------------------------------|------------------------------------------+
                                 | index_documents()
                                 v
+--------------------------------------------------------------------------+
|                       Local Filesystem                                   |
|  data/chroma_db/      <- ChromaDB persistent vector store               |
|  data/uploaded_docs/  <- user-uploaded .txt / .json files               |
|  data/processed/      <- all_chunks.json (reproducibility snapshot)     |
|  assets/              <- NUST Bank Excel + FAQ JSON (source of truth)    |
+--------------------------------------------------------------------------+
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Embeddings run **locally** | The 22M-param embedding model is tiny (~90 MB), fast on CPU, costs nothing per query, and keeps all knowledge-base content private |
| LLM runs via **Groq cloud API** | No GPU required; Groq's LPU hardware delivers ~200 ms responses; free tier covers demo and production loads |
| **ChromaDB** for vector storage | Embedded (no separate server), persistent across restarts, native cosine-similarity search, Python-native |
| **Lazy initialization** | The embedding model and LLM client load only on the first API request, so the server starts instantly |
| **Deterministic chunk IDs** | `{source}__chunk_{i}` format means re-uploading the same file updates existing vectors rather than creating duplicates |

---

## 3. Technology Stack

### Backend

| Layer | Technology | Version | Role |
|---|---|---|---|
| Web framework | FastAPI | >= 0.111 | REST API, request validation, CORS |
| ASGI server | Uvicorn | >= 0.30 | Async HTTP server |
| LLM inference | Groq API (Llama 3.2 3B) | — | Cloud-hosted, 3B params, free tier |
| Embedding model | sentence-transformers / all-MiniLM-L6-v2 | >= 2.2.2 | Local 22M-param bi-encoder |
| Vector database | ChromaDB | >= 1.0 | Persistent HNSW index, cosine similarity |
| Excel parsing | openpyxl | >= 3.1.2 | Reads multi-sheet .xlsx workbook |
| Data validation | Pydantic v2 | bundled with FastAPI | Request/response schemas |
| Language | Python | 3.10+ | All backend logic |

### Frontend

| Layer | Technology | Version | Role |
|---|---|---|---|
| Framework | Next.js 14 | 14.2.25 | SSR-ready React framework |
| UI library | React | 18.3.1 | Component rendering |
| Language | TypeScript | 5.5.4 | Type-safe frontend code |
| Styling | Custom CSS (globals.css) | — | Hand-written, no UI library dependency |

### Infrastructure / Data

| Component | Detail |
|---|---|
| Vector index algorithm | HNSW (Hierarchical Navigable Small World) — approximate nearest-neighbour |
| Similarity metric | Cosine similarity (ChromaDB `hnsw:space = cosine`) |
| Embedding dimensions | 384 (all-MiniLM-L6-v2 output size) |
| Chunk storage format | JSON documents in ChromaDB with `source`, `category`, `content` metadata |
| Persistence | ChromaDB writes to `data/chroma_db/` on every upsert |

---

## 4. Component Breakdown

### 4.1 `backend/app/core/settings.py` — Central Configuration

A frozen-dataclass singleton (`cfg`) that holds every tunable parameter in one place. Frozen means no part of the code can accidentally mutate settings at runtime.

```
AppConfig
 +-- PathSettings      -- all filesystem paths (assets, chroma, uploads, processed)
 +-- EmbeddingSettings -- model name, vector dimensions, ChromaDB collection name, batch size
 +-- LLMSettings       -- Groq model name, max_new_tokens, temperature
 +-- RetrieverSettings -- chunk_size, chunk_overlap, top_k, OOD thresholds
 +-- GuardrailSettings -- PII regex patterns, blocked topic keywords, jailbreak regex patterns
```

The `_PROJECT_ROOT` constant is resolved at import time using `Path(__file__).resolve().parents[3]`, making all paths absolute and correct regardless of the working directory the server is launched from.

### 4.2 `backend/app/core/guardrails.py` — Safety Layer

Stateless functions that inspect text and return a `GuardrailResult(is_safe, reason, filtered_text)` dataclass.

**Input guardrails** (called before any retrieval):
- `detect_jailbreak(text)` — matches 6 compiled regex patterns targeting prompt-injection phrases (e.g., "ignore all previous instructions", "you are now DAN", "reveal your system prompt")
- `detect_blocked_topics(text)` — substring-matches against ~16 sensitive keywords (passwords, PINs, CVVs, hacking instructions, etc.)
- `check_input_safety(text)` — runs both checks in order, returns the first failure

**Output guardrails** (called after LLM generation):
- `scrub_pii(text)` — applies 5 regex substitutions that replace PII with `[REDACTED]`: CNIC (`\d{5}-\d{7}-\d{1}`), Pakistani phone numbers, email addresses, 10-16 digit account numbers, credit card patterns
- `check_output_safety(text)` — runs PII scrub, then checks for patterns that suggest the model disclosed a password, CVV, or card/account number directly

### 4.3 `backend/app/core/llm_engine.py` — RAG Orchestrator

The `LLMEngine` class ties together all components and runs the 5-step pipeline on every chat query. It holds:
- A reference to the `EmbeddingStore` (injected at construction time by `RAGService`)
- A `Groq` API client (initialized with `GROQ_API_KEY` from the environment)

### 4.4 `backend/app/retrieval/embedding_store.py` — Vector Store Wrapper

Encapsulates:
1. The `SentenceTransformer` model (`all-MiniLM-L6-v2`) — loads once, held in memory
2. The `chromadb.PersistentClient` and the `nust_bank_knowledge` collection

Key methods:
- `index_documents(documents, batch_size=64)` — encodes texts in batches of 64, calls `collection.upsert()` with deterministic IDs
- `search(query, top_k=3)` — encodes the query vector, calls `collection.query()`, converts ChromaDB distances to similarities (`score = 1 - distance`)
- `document_count()` — fast `collection.count()` call, used by `/stats` and to detect an empty index on startup
- `reset()` — deletes and recreates the collection (used by `reset_chromadb.py`)

### 4.5 `backend/app/services/rag_service.py` — Service Layer

A thin dataclass that owns the singleton instances of `EmbeddingStore` and `LLMEngine` and is shared across all FastAPI requests.

**Lazy initialization pattern:**
```
First request to /chat or /upload
  -> _ensure_ready_store() checks if self._store is None
      -> creates EmbeddingStore (loads embedding model + opens ChromaDB)
      -> if ChromaDB is empty: runs load_all_documents() -> index_documents()
  -> _ensure_ready_engine() checks if self._engine is None
      -> creates LLMEngine (initializes Groq client)
```

This means server startup is instant. The embedding model load delay (~3 seconds) only happens on the very first query.

**Auto-recovery**: if ChromaDB initialization panics (corrupted files), the service catches the exception, wipes `data/chroma_db/`, and raises — the next request triggers a clean re-initialization.

### 4.6 `backend/app/main.py` — FastAPI Entrypoint

Defines 4 REST endpoints, applies CORS middleware (origin configurable via `FRONTEND_ORIGIN` environment variable), and holds the single `RAGService` instance at module level.

---

## 5. Chat Flow — Step by Step

This is exactly what happens inside the system from the moment the user clicks "Send" to the moment the answer appears on screen.

```
Browser              FastAPI            LLMEngine           Groq Cloud
   |                    |                   |                    |
   |-- POST /chat ----->|                   |                    |
   |  { "message" }     |                   |                    |
   |                    |-- service.chat()->|                    |
   |                    |                   |                    |
   |                    |          STEP 1: Input Guards          |
   |                    |          check_input_safety(query)     |
   |                    |          [if unsafe -> return now]     |
   |                    |                   |                    |
   |                    |          STEP 2: Retrieval             |
   |                    |          store.search(query, top_k=3)  |
   |                    |          [encode query -> HNSW search] |
   |                    |                   |                    |
   |                    |          STEP 3: OOD Check             |
   |                    |          cosine scores < threshold?    |
   |                    |          [if OOD -> return static msg] |
   |                    |                   |                    |
   |                    |          STEP 4: Groq API call ------->|
   |                    |          system + user messages        | Llama 3.2 3B
   |                    |          <-- raw_answer ---------------|
   |                    |                   |                    |
   |                    |          STEP 5: Output Guards         |
   |                    |          scrub_pii() + harmful check   |
   |                    |                   |                    |
   |<-- 200 JSON -------|<-- result dict ---|                    |
   | { answer,          |                   |                    |
   |   sources,         |                   |                    |
   |   latency_ms,      |                   |                    |
   |   guardrail_triggered,                 |                    |
   |   out_of_domain }  |                   |                    |
```

### Step 1 — Input Guardrails (< 1 ms, fully local)

The raw message text is passed to `check_input_safety()`. This runs two checks in sequence:

**1. Jailbreak detection** — the message is lowercased and matched against 6 compiled regular expressions:
- `r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)"`
- `r"(pretend|act|behave)\s+(as\s+if|like)\s+you\s+(are|have|can)"`
- `r"(disregard|forget|override)\s+(your|all|the)\s+(rules|guidelines|restrictions|instructions)"`
- `r"you\s+are\s+now\s+(DAN|unrestricted|unfiltered|jailbroken)"`
- `r"(reveal|show|tell\s+me|display)\s+(your|the)\s+(system\s+)?(prompt|instructions|rules)"`
- `r"(do\s+not|don'?t)\s+(follow|obey|listen\s+to)\s+(your|the)\s+(rules|guidelines)"`

**2. Blocked topic detection** — substring match against ~16 sensitive keywords including `"password"`, `"cvv"`, `"pin number"`, `"hack"`, `"exploit"`, `"bypass security"`.

If either check fails, the pipeline **immediately returns** the rejection reason as the answer — no retrieval, no API call, near-zero latency.

### Step 2 — Semantic Retrieval (local, ~5–50 ms on CPU)

`EmbeddingStore.search(query, top_k=3)` does the following:

1. The query string is fed through the `all-MiniLM-L6-v2` bi-encoder model, producing a single **384-dimensional float vector**.
2. ChromaDB executes an **HNSW approximate nearest-neighbour search** over all indexed document vectors using cosine distance.
3. The top-3 closest chunks are returned with their raw distance values.
4. Distances are converted to similarity scores: `score = 1.0 - distance`. A score of 1.0 means identical; 0.0 means completely dissimilar.
5. Each result is a dict: `{ content, source, category, score }`.

### Step 3 — Out-of-Domain Check (< 1 ms)

`is_out_of_domain(query, [score1, score2, score3])`:

- Computes the **maximum** and **average** of the 3 cosine similarity scores.
- If `max_score < 0.25` **AND** `avg_score < 0.20`, the query is declared out-of-domain.
- This means even the single best-matching chunk in the entire knowledge base barely resembles the query — a strong signal the question is not about NUST Bank.
- Returns a static, friendly "out of scope" message without making an API call.

In practice: a question about NUST Bank products typically scores 0.35–0.65. A question about the weather or a general topic scores 0.05–0.15. The dual-threshold prevents false positives on marginally relevant queries.

### Step 4 — Groq API Call (~150–400 ms)

`LLMEngine._generate(query, context_chunks)`:

The 3 retrieved chunks are concatenated with double newlines. Two messages are sent to the Groq API:

**System message** (sent on every request — sets persona and hard constraints):
```
You are NUST Bank's friendly and professional AI customer support assistant.

RULES:
1. Only answer questions related to NUST Bank products, services, accounts, and policies.
2. Base your answers STRICTLY on the provided context. Do NOT fabricate information.
... [7 rules total]
```

**User message** (carries the evidence and the question):
```
Answer the question using ONLY the context provided below.
Be specific, accurate, and helpful.
Use bullet points when listing multiple items.

Context:
[Chunk 1 text — ~500 characters]

[Chunk 2 text — ~500 characters]

[Chunk 3 text — ~500 characters]

Question: [user's original question]
```

Parameters: `temperature=0.3` (low randomness for factual consistency), `max_tokens=400`.

Using the **chat-completion format** (system + user roles) is significantly more reliable for instruction-following than flat text prompts. Llama 3.2 was instruction-tuned on this exact format.

### Step 5 — Output Guardrails (< 1 ms, local)

`check_output_safety(raw_answer)`:

1. `scrub_pii(text)` runs 5 regex substitutions over the model's response, replacing any leaked PII with `[REDACTED]`.
2. Two additional patterns check whether the model generated output that looks like it is directly disclosing a password, CVV, or account number (e.g., `"password is: abc123"`).
3. If harmful patterns are found after PII scrubbing, the safety response is returned instead of the model's output.
4. Otherwise the PII-scrubbed text is returned as the final answer.

### Final Response Shape

| Field | Type | Description |
|---|---|---|
| `answer` | string | Final answer text shown to the user |
| `sources` | list of strings | Unique source identifiers of the retrieved chunks |
| `latency_ms` | float | Total wall-clock time for the entire pipeline in milliseconds |
| `guardrail_triggered` | bool | True if input or output guardrail fired |
| `out_of_domain` | bool | True if OOD check rejected the query |

---

## 6. Document Upload & Ingestion Flow

When a user uploads a file through the web interface, the following happens:

```
Browser                      FastAPI                Filesystem + ChromaDB
   |                             |                          |
   |-- POST /upload (multipart)->|                          |
   |   file.txt or file.json     |                          |
   |                             | validate extension       |
   |                             |-- write file ----------->|
   |                             |  data/uploaded_docs/     |
   |                             |                          |
   |                             | RAGService.upload_and_index()
   |                             |                          |
   |                             | ingest_uploaded_documents()
   |                             |  +-- read file           |
   |                             |  +-- clean_text()        |
   |                             |  +-- anonymize_text()    |
   |                             |  +-- -> raw document     |
   |                             |                          |
   |                             | chunk_documents()        |
   |                             |  +-- chunk_text() x N   |
   |                             |  +-- assign chunk_ids   |
   |                             |  +-- -> chunk list      |
   |                             |                          |
   |                             | store.index_documents()  |
   |                             |  +-- encode embeddings  |
   |                             |  +-- collection.upsert()>|
   |                             |                          |
   |<-- 200 { filename, ---------|                          |
   |    indexed_chunks,          |                          |
   |    indexed_documents_total }|                          |
```

### Initial Knowledge-Base Ingestion (runs on first server startup if ChromaDB is empty)

```
load_all_documents()
  |
  +-- ingest_excel(assets/NUST Bank-Product-Knowledge.xlsx)
  |     for each worksheet:
  |       extract all cell values
  |       join cell values into rows
  |       join rows into one document per sheet
  |       clean_text() -> anonymize_text()
  |       -> document: { source, category, content }
  |
  +-- ingest_faq_json(assets/funds_transfer_app_features_faq.json)
  |     for each category -> each Q&A pair:
  |       format as "Question: ...\nAnswer: ..."
  |       clean_text() -> anonymize_text()
  |       -> document: { source, category, content }
  |
  +-- ingest_uploaded_documents(data/uploaded_docs/)
  |     for each .txt:  read -> clean -> anonymize -> document
  |     for each .json with "categories" key: delegate to ingest_faq_json()
  |     for other .json: json.dumps -> clean -> anonymize -> document
  |
  +-- chunk_documents(all raw docs)
  |     split each document into overlapping 500-char chunks
  |     assign deterministic chunk_ids
  |
  +-- persist to data/processed/all_chunks.json  (reproducibility snapshot)
  |
  +-- EmbeddingStore.index_documents(chunks)
        encode all chunks in batches of 64
        upsert into ChromaDB with deterministic IDs
```

---

## 7. Chunking Algorithm — In Detail

Chunking splits long documents into small, retrievable passages. The algorithm is in `backend/app/ingestion/chunker.py`.

### Parameters

| Parameter | Value | Purpose |
|---|---|---|
| `chunk_size` | 500 characters | Target length of each chunk |
| `chunk_overlap` | 50 characters | Characters shared between consecutive chunks |

### Why 500-character chunks?

- Short enough to fit 3 chunks in a single LLM prompt without overloading the context
- Long enough to contain a complete thought or policy statement (roughly 3–6 sentences)
- The `all-MiniLM-L6-v2` model was trained on sentence-level semantics — this length is an ideal match

### The `chunk_text()` Algorithm (sentence-aware)

```
Input: full document text (possibly thousands of characters)

1. If len(text) <= chunk_size  ->  return [text] as a single chunk (no split needed)

2. start = 0
3. While start < len(text):
   a. end = start + chunk_size                   <- candidate end position
   b. chunk = text[start:end]                    <- candidate chunk window
   c. last_period  = chunk.rfind(".")             <- last "." in this window
   d. last_newline = chunk.rfind("\n")            <- last newline in this window
   e. break_pos = max(last_period, last_newline)  <- prefer natural boundary
   f. If break_pos > chunk_size // 2:            <- boundary is in second half?
        end = start + break_pos + 1              <- snap end to sentence boundary
        chunk = text[start:end]                  <- re-slice to natural end
   g. Append chunk.strip() to result list
   h. start = end - chunk_overlap                <- back up by 50 chars for overlap

4. Filter out empty strings -> return chunk list
```

**Concrete example** with chunk_size=500, overlap=50:

```
Document: 1200 characters of text

Chunk 1: characters   0 to ~490  (snapped to last period before char 500)
Chunk 2: characters ~440 to ~930  (starts 50 chars before chunk 1 ended)
Chunk 3: characters ~880 to 1200  (final remainder)
```

The **50-character overlap** is critical: it ensures that a sentence that spans a chunk boundary is captured fully in at least one chunk, preventing factual fragmentation across chunk edges.

### Chunk ID Assignment

After `chunk_text()` splits a document's content string, `chunk_documents()` builds a metadata-rich dict for each chunk:

```python
{
    "source":   "NUST_Bank_Product_Knowledge.xlsx/Sahar Account",
    "category": "Sahar Account",
    "content":  "The NUST Sahar Account is a Shariah-compliant savings...",
    "chunk_id": "NUST_Bank_Product_Knowledge.xlsx/Sahar Account__chunk_0"
}
```

The `chunk_id` follows the format `{source}__chunk_{i}`. This is the **primary key** used in ChromaDB upserts. Because IDs are deterministic and stable, re-ingesting the same source file always updates the same vectors in place rather than duplicating them.

---

## 8. Embedding & Vector Store

### The Embedding Model: `all-MiniLM-L6-v2`

| Property | Value |
|---|---|
| Architecture | Bi-encoder (BERT-based, 6 transformer layers) |
| Parameters | 22 million |
| Output vector size | 384 dimensions (float32) |
| Training objective | Contrastive learning (mean pooling + cosine similarity) |
| Training data | 1 billion+ sentence pairs from diverse domains |
| Model size on disk | ~90 MB |
| Inference speed | ~5–15 ms per batch of 64 sentences on CPU |

A **bi-encoder** means both the document text and the query text are encoded independently into the same 384-dimensional vector space. Semantically similar texts land close together in this space. This architecture enables pre-computing all document vectors once at indexing time and retrieving in milliseconds at query time.

### ChromaDB Collection

Each chunk is stored as a record with four components:

| Component | Content |
|---|---|
| **ID** | Deterministic `chunk_id` string (e.g., `FAQ_JSON/Accounts/Q_3__chunk_0`) |
| **Embedding** | 384-element float32 vector |
| **Document** | Raw chunk text (stored for retrieval, returned in search results) |
| **Metadata** | `{ "source": str, "category": str }` |

The collection is configured with `{"hnsw:space": "cosine"}`:
- All distances are **cosine distances** (not Euclidean)
- Cosine distance = `1 - cosine_similarity`, so distance 0 = identical vectors, distance 1 = orthogonal (completely unrelated)
- Cosine distance is preferred over Euclidean for high-dimensional text embeddings because it is invariant to vector magnitude

### HNSW Index

HNSW (Hierarchical Navigable Small World) is a graph-based approximate nearest-neighbour algorithm used internally by ChromaDB. Instead of comparing the query vector against every stored vector (brute force O(N)), it navigates a layered graph structure to find approximate nearest neighbours in O(log N) time. At the scale of ~1,000–5,000 chunks, query times are sub-millisecond.

---

## 9. Retrieval Mechanism

```python
def search(query: str, top_k: int = 3) -> List[Dict]:
    query_vec = model.encode([query]).tolist()      # 1 x 384 float vector
    results = collection.query(
        query_embeddings=query_vec,
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    # Convert ChromaDB cosine distance to similarity score
    for doc, meta, dist in zip(...):
        score = round(1.0 - float(dist), 4)
        output.append({ content, source, category, score })
    return output
```

### Out-of-Domain Thresholds

| Threshold | Value | Meaning |
|---|---|---|
| `ood_max_score_threshold` | 0.25 | If even the single best match scores below 0.25... |
| `ood_avg_score_threshold` | 0.20 | ...AND the average of all 3 scores is below 0.20... |
| Combined | Both must be true | ...the query is declared out-of-domain |

Both conditions must be simultaneously true to avoid false positives. A genuinely on-topic query almost always scores at least one chunk above 0.30. An off-topic query (weather, sports, general knowledge) typically scores all chunks below 0.15.

---

## 10. Guardrail System

### Input Guardrail Patterns

**Jailbreak patterns** (6 regex patterns, case-insensitive):
```
ignore (all) (previous|prior|above) (instructions|prompts|rules)
(pretend|act|behave) (as if|like) you (are|have|can)
(disregard|forget|override) (your|all|the) (rules|guidelines|restrictions|instructions)
you are now (DAN|unrestricted|unfiltered|jailbroken)
(reveal|show|tell me|display) (your|the) (system) (prompt|instructions|rules)
(do not|don't) (follow|obey|listen to) (your|the) (rules|guidelines)
```

**Blocked topics** (16 substring keywords):
`password`, `pin number`, `cvv`, `secret question`, `social security`, `hack`, `exploit`, `bypass security`, `ignore previous instructions`, `ignore above`, `disregard your instructions`, `pretend you are`, `act as if you have no restrictions`, `reveal system prompt`, `show me your prompt`, `what are your instructions`, `repeat your system message`

### PII Detection Patterns

Used both during **ingestion** (anonymize source data before storage) and **output** (scrub model responses before display):

| PII Type | Regex Pattern | Example |
|---|---|---|
| CNIC | `\b\d{5}-\d{7}-\d{1}\b` | `42101-1234567-8` |
| Pakistani phone | `\b(?:\+92\|0)\s*\d{3}[\s-]?\d{7}\b` | `0321-1234567` |
| Email address | RFC-style pattern | `user@example.com` |
| Bank account number | `\b\d{10,16}\b` | Any 10–16 digit sequence |
| Credit card | `\b(?:\d{4}[\s-]?){3}\d{4}\b` | `1234 5678 9012 3456` |

**Important distinction**: During **ingestion**, PII is replaced with labelled placeholders like `[REDACTED_CNIC]` so the knowledge base itself never stores raw PII. During **output scrubbing**, PII is replaced with the generic `[REDACTED]`.

### Harmful Output Patterns

Two additional patterns check the model's generated text for specific disclosure:
```
(password|cvv|pin) (is|:) \S+
(account number|card number) (is|:) \d+
```

---

## 11. LLM — Groq API (Llama 3.2 3B)

### Why Groq?

| Factor | Detail |
|---|---|
| **Speed** | Groq's LPU (Language Processing Unit) delivers ~150–400 ms total latency for 3B models — faster than most GPU providers |
| **Cost** | Free tier: 14,400 requests/day, 30 requests/minute — more than sufficient for demos and light production |
| **Model size** | `llama-3.2-3b-preview` has exactly 3 billion parameters |
| **Quality** | Meta's Llama 3.2 3B instruction-tuned model is significantly more capable at instruction-following and reasoning than same-size seq2seq models |
| **No local resources** | No GPU, no large RAM requirement, no multi-GB model download — just an API key |

### Model: Meta Llama 3.2 3B Instruct

| Property | Value |
|---|---|
| Architecture | Decoder-only transformer (Llama 3 family) |
| Parameters | 3,213,000,000 (3.2 billion) |
| Context window | 128,000 tokens |
| Training | Instruction-tuned with RLHF and preference data |
| Key capability | Strong instruction following, factual grounding, multi-turn chat |

### Generation Parameters

| Parameter | Value | Effect |
|---|---|---|
| `temperature` | 0.3 | Low randomness — answers are consistent and factual rather than creative |
| `max_tokens` | 400 | Limits response length; prevents verbose or runaway generation |
| `model` | `llama-3.2-3b-preview` | Groq-hosted 3B model identifier |

---

## 12. API Reference

### `GET /health`

Returns service status. Always responds quickly regardless of initialization state.

**Response:**
```json
{ "status": "ok", "service": "nust-bank-backend" }
```

### `GET /stats`

Returns the current state of the knowledge base and model configuration. Triggers lazy initialization of the EmbeddingStore on first call.

**Response:**
```json
{
  "indexed_documents": 1247,
  "llm_model": "llama-3.2-3b-preview",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

### `POST /chat`

Main chat endpoint. Runs the full 5-step RAG pipeline.

**Request body:**
```json
{ "message": "What is the daily transfer limit on NUST mobile banking?" }
```
Constraint: `message` must be 1–4000 characters.

**Response:**
```json
{
  "answer": "The daily transfer limit on NUST Bank mobile banking is Rs. 500,000 per day...",
  "sources": [
    "FAQ_JSON/Funds Transfer/Q_3",
    "NUST_Bank_Product_Knowledge.xlsx/Mobile Banking"
  ],
  "latency_ms": 387.4,
  "guardrail_triggered": false,
  "out_of_domain": false
}
```

### `POST /upload`

Accepts a `.txt` or `.json` file, saves it to `data/uploaded_docs/`, ingests and indexes it immediately.

**Request:** `multipart/form-data` with a `file` field (`.txt` or `.json` only).

**Response:**
```json
{
  "filename": "new_policy.txt",
  "indexed_chunks": 14,
  "indexed_documents_total": 1261
}
```

Returns `400` for unsupported file types, `500` for processing failures.

---

## 13. Project Structure

```
LLM_Project/
|
+-- backend/
|   +-- .env.example            <- copy to .env and fill GROQ_API_KEY
|   +-- app/
|       +-- __init__.py
|       +-- main.py             <- FastAPI app, 4 endpoints, CORS middleware
|       +-- schemas.py          <- Pydantic request/response models
|       |
|       +-- core/
|       |   +-- settings.py     <- frozen-dataclass config singleton (cfg)
|       |   +-- guardrails.py   <- jailbreak, blocked topics, PII scrubbing
|       |   +-- llm_engine.py   <- 5-step RAG pipeline + Groq API client
|       |   +-- prompt_engine.py<- SYSTEM_PROMPT, static response strings
|       |
|       +-- ingestion/
|       |   +-- text_cleaner.py <- normalize whitespace, anonymize PII
|       |   +-- excel_loader.py <- parse multi-sheet Excel workbook
|       |   +-- json_loader.py  <- parse FAQ JSON (categories -> Q&A pairs)
|       |   +-- upload_loader.py<- read uploaded .txt and .json files
|       |   +-- chunker.py      <- sentence-aware overlapping chunk splitter
|       |   +-- pipeline.py     <- orchestrates all loaders + chunker
|       |
|       +-- retrieval/
|       |   +-- embedding_store.py <- SentenceTransformer + ChromaDB wrapper
|       |
|       +-- services/
|           +-- rag_service.py  <- lazy-init singleton, bridges API <-> engine
|
+-- frontend/
|   +-- app/
|   |   +-- layout.tsx          <- root HTML shell
|   |   +-- page.tsx            <- chat UI + upload modal (main component)
|   |   +-- globals.css         <- all styling
|   +-- lib/
|   |   +-- api.ts              <- typed fetch wrappers for all 4 endpoints
|   +-- package.json
|
+-- assets/
|   +-- NUST Bank-Product-Knowledge.xlsx   <- primary knowledge source (Excel)
|   +-- funds_transfer_app_features_faq.json <- FAQ knowledge source (JSON)
|
+-- data/                       <- generated at runtime (git-ignored)
|   +-- chroma_db/              <- persistent ChromaDB vector index
|   +-- processed/              <- all_chunks.json reproducibility snapshot
|   +-- uploaded_docs/          <- user-uploaded files
|
+-- scripts/
|   +-- fine_tune.py            <- LoRA fine-tuning script (offline, not runtime)
|
+-- reset_chromadb.py           <- wipe and recreate chroma_db/ (recovery tool)
+-- requirements.txt
+-- README.md
```

---

## 14. Configuration Reference

All settings are in [backend/app/core/settings.py](backend/app/core/settings.py) under the `cfg` singleton.

### LLM Settings

| Setting | Default | Description |
|---|---|---|
| `model_name` | `llama-3.2-3b-preview` | Groq model identifier |
| `max_new_tokens` | `400` | Maximum tokens in the generated response |
| `temperature` | `0.3` | Sampling temperature (0 = fully deterministic, 1 = creative) |

Available Groq models under 6B parameters:
- `llama-3.2-1b-preview` — 1B params, fastest inference
- `llama-3.2-3b-preview` — 3B params, recommended balance of speed and quality

### Embedding Settings

| Setting | Default | Description |
|---|---|---|
| `model_name` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace model ID |
| `dimension` | `384` | Output vector size (fixed by the model architecture) |
| `collection_name` | `nust_bank_knowledge` | ChromaDB collection name |
| `batch_size` | `64` | Sentences encoded per CPU batch during indexing |

### Retriever Settings

| Setting | Default | Description |
|---|---|---|
| `chunk_size` | `500` | Max characters per chunk |
| `chunk_overlap` | `50` | Overlap characters between consecutive chunks |
| `top_k` | `3` | Number of chunks retrieved per query |
| `ood_max_score_threshold` | `0.25` | Max cosine score below which query is out-of-domain |
| `ood_avg_score_threshold` | `0.20` | Avg cosine score below which query is out-of-domain |

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | **Yes** | API key from console.groq.com (free tier available) |
| `FRONTEND_ORIGIN` | No (default: `http://localhost:3000`) | CORS allowed origin for the Next.js frontend |

---

## 15. Setup & Running

### Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- A free Groq API key from console.groq.com

### Step 1 — Install Python dependencies

```bash
# Windows
venv\Scripts\pip install -r requirements.txt

# macOS / Linux
venv/bin/pip install -r requirements.txt
```

On first run, `sentence-transformers` downloads `all-MiniLM-L6-v2` (~90 MB) from HuggingFace automatically. This is the only file downloaded — the LLM runs remotely on Groq.

### Step 2 — Configure the Groq API key

```bash
# Copy the example environment file
copy backend\.env.example backend\.env

# Open backend\.env and set your key:
# GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Get your free key at console.groq.com (no credit card required, 14,400 requests/day free).

### Step 3 — Start the FastAPI backend

```bash
cd D:\LLM_Project
venv\Scripts\uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

On the **first request**, the system:
1. Loads the embedding model (~3 seconds, one-time)
2. Opens ChromaDB
3. If ChromaDB is empty: ingests and indexes the Excel + FAQ knowledge base (~30–90 seconds)

All subsequent requests skip the initialization entirely.

### Step 4 — Start the Next.js frontend

```bash
cd frontend
npm install        # first time only
npm run dev
```

Open **http://localhost:3000**.

### Step 5 — Verify

```bash
# Health check
curl http://localhost:8000/health

# Check indexed chunk count
curl http://localhost:8000/stats

# Send a test query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What is the NUST Sahar Account?\"}"
```

### Resetting ChromaDB (if corrupted)

```bash
python reset_chromadb.py
# Restart the backend — it will re-index automatically on first request
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| `GROQ_API_KEY not set` | Missing environment variable | Create `backend/.env` from `backend/.env.example` and add your key |
| `No module named 'groq'` | Dependencies not installed | Run `pip install -r requirements.txt` |
| `pyo3_runtime.PanicException` | Corrupted ChromaDB files | Run `python reset_chromadb.py` then restart backend |
| Backend returns OOD for valid questions | Initial indexing did not complete | Check server logs; run `GET /stats` to verify chunk count > 0 |
| `No module named 'backend'` | Not running from project root | Always `cd LLM_Project` before starting uvicorn |
| Port 8000 already in use | Previous backend process still running | `Get-Process -Name python \| Stop-Process` (Windows PowerShell) |
