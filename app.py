# ============================================================
# app.py — Main Streamlit Entry Point
# ============================================================


import streamlit as st
from streamlit import session_state as ss

from components.sidebar import render_sidebar
from components.chat import render_chat
import config

# ── Page Configuration ───────────────────────────────────────
# Must be the FIRST Streamlit call in the script
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton > button {
        border-radius: 8px;
    }
    .stChatMessage {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialisation ─────────────────────────────
# Set default values for all keys on first run.
# if key already exists → skip (don't overwrite live data)
def init_session_state():
    defaults = {
        "chat_history":       [],    # list of {role, content, sources}
        "vectorstore":        None,  # ChromaDB Chroma object
        "rag_chain":          None,  # ConversationalRetrievalChain
        "uploaded_filenames": [],    # track indexed filenames
        "total_chunks":       0,     # total chunks indexed
        "llm_provider":       "OpenAI GPT-4o",
    }
    for key, val in defaults.items():
        if key not in ss:
            ss[key] = val

init_session_state()


# ── Header ───────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>{config.APP_TITLE}</h1>
    <p>{config.APP_SUBTITLE}</p>
</div>
""", unsafe_allow_html=True)


# ── Render Components ────────────────────────────────────────
# Sidebar handles uploads + settings
render_sidebar()

# Main area handles chat
render_chat()


# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>Built with LangChain · ChromaDB · OpenAI / Gemini · Streamlit</small></center>",
    unsafe_allow_html=True,
)
