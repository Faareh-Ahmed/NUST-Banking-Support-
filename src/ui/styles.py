"""
src/ui/styles.py
-----------------
Streamlit custom CSS injected once at app startup.
"""

import streamlit as st


_CSS = """
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
"""


def inject_styles() -> None:
    """Inject the application-wide CSS into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)
