"""Document ingestion sub-package: loading, cleaning, chunking, pipeline."""

from src.ingestion.text_cleaner import clean_text, anonymize_text
from src.ingestion.excel_loader import ingest_excel
from src.ingestion.json_loader import ingest_faq_json
from src.ingestion.upload_loader import ingest_uploaded_documents
from src.ingestion.chunker import chunk_text, chunk_documents
from src.ingestion.pipeline import load_all_documents

__all__ = [
    "clean_text",
    "anonymize_text",
    "ingest_excel",
    "ingest_faq_json",
    "ingest_uploaded_documents",
    "chunk_text",
    "chunk_documents",
    "load_all_documents",
]
