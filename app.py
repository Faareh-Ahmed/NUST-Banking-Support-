"""
app.py
-------
NUST Bank AI Customer Support — Streamlit entry point.

This file is intentionally thin: it wires together the sub-packages
from ``src/`` and delegates all rendering / logic to them.
"""

import streamlit as st

# ── Page config must be the first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="NUST Bank AI Support",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.ui.styles import inject_styles
from src.ui.sidebar import render_sidebar
from src.ui.chat import init_chat_state, render_chat


@st.cache_resource(show_spinner="Loading AI models… This may take a minute on first run.")
def _load_engine():
    """
    Load the embedding store and LLM engine once and cache them across
    Streamlit reruns / sessions.
    """
    from src.retrieval.embedding_store import EmbeddingStore
    from src.ingestion.pipeline import load_all_documents
    from src.core.llm_engine import LLMEngine

    store = EmbeddingStore()
    if store.document_count() == 0:
        with st.spinner("Indexing NUST Bank knowledge base…"):
            docs = load_all_documents()
            store.index_documents(docs)

    engine = LLMEngine(embedding_store=store)
    return engine, store


def main() -> None:
    inject_styles()
    init_chat_state()

    engine, store = _load_engine()

    render_sidebar(store)
    render_chat(engine)


if __name__ == "__main__":
    main()
