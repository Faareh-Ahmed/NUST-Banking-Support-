"""
src/ui/sidebar.py
------------------
Streamlit sidebar: system stats and document-upload panel.
"""

import os
import streamlit as st

from src.core.settings import cfg
from src.retrieval.embedding_store import EmbeddingStore


def render_sidebar(store: EmbeddingStore) -> None:
    """Render the complete sidebar for the given *store* instance."""
    with st.sidebar:
        st.markdown("## 🏦 NUST Bank AI Support")
        st.markdown("---")

        _render_stats(store)
        st.markdown("---")
        _render_upload_panel(store)
        st.markdown("---")
        _render_about()


# ── Private helpers ───────────────────────────────────────────────────────────

def _render_stats(store: EmbeddingStore) -> None:
    st.markdown("### 📊 System Status")
    st.metric("Indexed Documents", store.document_count())


def _render_upload_panel(store: EmbeddingStore) -> None:
    st.markdown("### 📄 Upload New Documents")
    st.markdown(
        "Add new FAQs, policies, or product info. "
        "Supported formats: `.txt`, `.json`"
    )

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "json"],
        help="Upload a text or JSON file to add to the knowledge base.",
    )

    if uploaded_file is not None and st.button("📥 Add to Knowledge Base", type="primary"):
        save_path = os.path.join(cfg.paths.uploaded_docs_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Indexing new document..."):
            from src.ingestion.upload_loader import ingest_uploaded_documents
            from src.ingestion.chunker import chunk_documents

            new_docs = ingest_uploaded_documents()
            new_chunks = chunk_documents(new_docs)
            if new_chunks:
                store.index_documents(new_chunks)
                st.success(
                    f"✅ Added **{uploaded_file.name}** — "
                    f"{len(new_chunks)} chunks indexed!"
                )
            else:
                st.warning("File was empty or could not be parsed.")


def _render_about() -> None:
    st.markdown("### ℹ️ About")
    st.markdown(
        "This AI assistant uses **Retrieval-Augmented Generation (RAG)** "
        "with an open-source LLM to answer your banking questions based "
        "on NUST Bank's official knowledge base."
    )
    st.markdown(
        "**Model:** Flan-T5-XL  \n"
        "**Embeddings:** MiniLM-L6-v2  \n"
        "**Vector Store:** ChromaDB"
    )
    st.caption("⚠️ This is an AI assistant. For critical matters, contact the bank directly.")
