"""
NUST Bank AI Customer Support — Streamlit Application
------------------------------------------------------
Provides a chat interface for customers to ask banking questions,
and an admin panel to upload new documents for real-time updates.
"""

import os
import json
import time
import streamlit as st

from config import UPLOADED_DOCS_DIR, PROCESSED_DIR

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NUST Bank AI Support",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a5276;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #5d6d7e;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-tag {
        background: #eaf2f8;
        color: #2e86c1;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 4px;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI models... This may take a minute on first run.")
def load_engine():
    """Load the LLM engine (cached across sessions)."""
    from embedding_store import EmbeddingStore
    from data_ingestion import load_all_documents
    from llm_engine import LLMEngine

    store = EmbeddingStore()
    if store.document_count() == 0:
        with st.spinner("Indexing NUST Bank knowledge base..."):
            docs = load_all_documents()
            store.index_documents(docs)

    engine = LLMEngine(embedding_store=store)
    return engine, store


def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Welcome to NUST Bank AI Support! 👋\n\n"
                    "I can help you with questions about:\n"
                    "- **Account types** (Sahar, Freelancer Digital, etc.)\n"
                    "- **Finance products** (Auto, Personal, Mortgage)\n"
                    "- **Mobile app** features & funds transfer\n"
                    "- **Credit/Debit cards**\n"
                    "- **Insurance** products\n\n"
                    "How can I assist you today?"
                ),
            }
        ]


# ── Sidebar: Document Upload & Info ──────────────────────────────────────────

def render_sidebar(store):
    with st.sidebar:
        st.markdown("## 🏦 NUST Bank AI Support")
        st.markdown("---")

        # Stats
        st.markdown("### 📊 System Status")
        doc_count = store.document_count()
        st.metric("Indexed Documents", doc_count)

        st.markdown("---")

        # Document Upload
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

        if uploaded_file is not None:
            if st.button("📥 Add to Knowledge Base", type="primary"):
                # Save file
                save_path = os.path.join(UPLOADED_DOCS_DIR, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Re-index
                with st.spinner("Indexing new document..."):
                    from data_ingestion import (
                        ingest_uploaded_documents,
                        chunk_documents,
                    )

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

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "This AI assistant uses **Retrieval-Augmented Generation (RAG)** "
            "with an open-source LLM to answer your banking questions based "
            "on NUST Bank's official knowledge base."
        )
        st.markdown(
            "**Model:** Flan-T5-Base  \n"
            "**Embeddings:** MiniLM-L6-v2  \n"
            "**Vector Store:** NumPy + JSON"
        )
        st.caption("⚠️ This is an AI assistant. For critical matters, contact the bank directly.")


# ── Main Chat Interface ─────────────────────────────────────────────────────

def render_chat(engine):
    st.markdown('<div class="main-header">NUST Bank AI Customer Support</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Ask anything about NUST Bank products & services</div>',
        unsafe_allow_html=True,
    )

    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                source_html = " ".join(
                    f'<span class="source-tag">{s}</span>' for s in msg["sources"]
                )
                st.markdown(f"**Sources:** {source_html}", unsafe_allow_html=True)
            if "latency" in msg:
                st.caption(f"⏱️ Response time: {msg['latency']}ms")

    # User input
    if user_input := st.chat_input("Type your question here..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = engine.answer(user_input)

            st.markdown(response["answer"])

            if response["sources"]:
                source_html = " ".join(
                    f'<span class="source-tag">{s}</span>' for s in response["sources"]
                )
                st.markdown(f"**Sources:** {source_html}", unsafe_allow_html=True)

            if response["guardrail_triggered"]:
                st.warning("⚠️ Safety filter was applied to this response.")

            st.caption(f"⏱️ Response time: {response['latency_ms']}ms")

        # Save to history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response["answer"],
                "sources": response["sources"],
                "latency": response["latency_ms"],
            }
        )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    init_session()
    engine, store = load_engine()
    render_sidebar(store)
    render_chat(engine)


if __name__ == "__main__":
    main()
