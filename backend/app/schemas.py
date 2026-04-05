"""Pydantic schemas for backend API contracts."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float
    guardrail_triggered: bool
    out_of_domain: bool


class UploadResponse(BaseModel):
    filename: str
    indexed_chunks: int
    indexed_documents_total: int


class StatsResponse(BaseModel):
    indexed_documents: int
    llm_model: str
    embedding_model: str


class HealthResponse(BaseModel):
    status: str
    service: str
