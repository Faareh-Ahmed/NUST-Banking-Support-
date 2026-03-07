"""
src/ui/chat.py
---------------
Streamlit main chat interface: message history display and user input handling.
"""

import streamlit as st

from src.core.llm_engine import LLMEngine


# ── Welcome message shown on the first load ───────────────────────────────────
_WELCOME_MESSAGE = (
    "Welcome to NUST Bank AI Support! 👋\n\n"
    "I can help you with questions about:\n"
    "- **Account types** (Sahar, Freelancer Digital, etc.)\n"
    "- **Finance products** (Auto, Personal, Mortgage)\n"
    "- **Mobile app** features & funds transfer\n"
    "- **Credit/Debit cards**\n"
    "- **Insurance** products\n\n"
    "How can I assist you today?"
)


def init_chat_state() -> None:
    """Seed ``st.session_state.messages`` with the welcome message."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": _WELCOME_MESSAGE}
        ]


def render_chat(engine: LLMEngine) -> None:
    """Render the full chat panel: header, history, and input box."""
    st.markdown(
        '<div class="main-header">NUST Bank AI Customer Support</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Ask anything about NUST Bank products & services</div>',
        unsafe_allow_html=True,
    )

    _render_message_history()
    _handle_user_input(engine)


# ── Private helpers ───────────────────────────────────────────────────────────

def _render_source_tags(sources: list) -> None:
    """Render coloured source tags below a message."""
    if sources:
        tags_html = " ".join(
            f'<span class="source-tag">{s}</span>' for s in sources
        )
        st.markdown(f"**Sources:** {tags_html}", unsafe_allow_html=True)


def _render_message_history() -> None:
    """Display all messages stored in the session."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            _render_source_tags(msg.get("sources", []))
            if "latency" in msg:
                st.caption(f"⏱️ Response time: {msg['latency']}ms")


def _handle_user_input(engine: LLMEngine) -> None:
    """Accept user input, generate a response, and update the history."""
    user_input = st.chat_input("Type your question here...")
    if not user_input:
        return

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = engine.answer(user_input)

        st.markdown(response["answer"])
        _render_source_tags(response["sources"])

        if response["guardrail_triggered"]:
            st.warning("⚠️ Safety filter was applied to this response.")

        st.caption(f"⏱️ Response time: {response['latency_ms']}ms")

    # Persist assistant message in history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response["sources"],
        "latency": response["latency_ms"],
    })
