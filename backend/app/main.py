"""FastAPI entrypoint for NUST Bank AI Support backend."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure absolute imports work even when running from `backend/app`
# e.g. `uvicorn main:app` with CWD `backend/app`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    StatsResponse,
    UploadResponse,
)
from backend.app.services.rag_service import RAGService
from backend.app.core.settings import cfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

service = RAGService()

app = FastAPI(
    title="NUST Bank AI Support API",
    version="1.0.0",
    description="Backend API for the Next.js customer support interface.",
)

frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        return HealthResponse(**service.health())
    except Exception as exc:
        logger.exception("/health failed: %s", exc)
        return HealthResponse(status="error", service="nust-bank-backend")


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    try:
        return StatsResponse(**service.stats())
    except Exception as exc:
        logger.exception("/stats failed: %s", exc)
        return StatsResponse(
            indexed_documents=0,
            llm_model=cfg.llm.model_name,
            embedding_model=cfg.embedding.model_name,
        )


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        result = service.chat(payload.message)
        return ChatResponse(**result)
    except Exception as exc:  # pragma: no cover - defensive API boundary
        logger.exception("/chat failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate answer.")


@app.post("/upload", response_model=UploadResponse)
def upload(file: UploadFile = File(...)) -> UploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".json"}:
        raise HTTPException(status_code=400, detail="Only .txt and .json files are supported.")

    os.makedirs(cfg.paths.uploaded_docs_dir, exist_ok=True)
    save_path = Path(cfg.paths.uploaded_docs_dir) / (file.filename or "uploaded_file")

    try:
        with open(save_path, "wb") as f:
            f.write(file.file.read())

        result = service.upload_and_index(file.filename or save_path.name)
        return UploadResponse(**result)
    except Exception as exc:  # pragma: no cover - defensive API boundary
        logger.exception("/upload failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to upload and index document.")
    finally:
        file.file.close()
