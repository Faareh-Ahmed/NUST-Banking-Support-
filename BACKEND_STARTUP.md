# Backend Startup Guide

## If you're seeing ChromaDB panics or corruption errors:

### Step 1: Reset ChromaDB
```bash
cd D:\LLM_Project
python reset_chromadb.py
```

This will:
- Delete the corrupted `data/chroma_db/` directory
- Create a fresh, empty ChromaDB directory
- On first backend startup, the knowledge base will be re-indexed automatically

### Step 2: Start the Backend

**From project root** (recommended):
```bash
cd D:\LLM_Project
venv\Scripts\uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

**Alternative (if running from `backend/app`)**:
```bash
cd D:\LLM_Project\backend\app
venv\Scripts\uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Step 3: Test the Backend

Once the backend is running, test these endpoints:

#### Health Check
```bash
curl http://localhost:8000/health
```
Expected response:
```json
{"status": "ok", "service": "nust-bank-backend"}
```

#### Stats (indexed documents)
```bash
curl http://localhost:8000/stats
```
Expected response:
```json
{
  "indexed_documents": 1000,
  "llm_model": "google/flan-t5-small",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

#### Chat Query
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the daily transfer limit?"}'
```

#### File Upload
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@path/to/document.txt"
```

---

## What Changed

The backend now has:
- ✅ **Safe ChromaDB initialization** — catches panics, logs errors, auto-resets on corruption
- ✅ **Graceful error handling** — `/stats` and `/health` never crash the server
- ✅ **Project-root sys.path fix** — works from any working directory
- ✅ **Helper reset script** — one-command database recovery

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `pyo3_runtime.PanicException` | Run `python reset_chromadb.py` then restart backend |
| `No module named 'backend'` | Make sure you're running from project root, or uvicorn is in venv |
| `/stats` returns 500 error | Check `reset_chromadb.py` output; ChromaDB may need cleanup |
| Port 8000 already in use | Kill existing process: `Get-Process -Name python \| Stop-Process` (Windows) |

