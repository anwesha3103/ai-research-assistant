# ============================================================
# components/sidebar.py — Sidebar UI Component
# ============================================================
# The sidebar handles:
#   • LLM provider switching
#   • Document uploading + indexing
#   • Displaying indexed files + stats
#   • Clearing everything

import streamlit as st
from streamlit import session_state as ss

from components.uploader import render_uploader
from rag.loader import load_documents, chunk_documents, get_document_stats
from rag.vectorstore import (
    build_vectorstore,
    add_to_vectorstore,
    clear_vectorstore,
    get_retriever,
)
from rag.chain import get_llm, build_rag_chain
from utils.logger import get_logger
import config

logger = get_logger()


def render_sidebar():
    """
    Renders the full sidebar and handles all document
    ingestion logic. Returns the selected LLM provider.
    """
    with st.sidebar:
        st.markdown("##  Settings")

        # ── LLM Provider Switcher ────────────────────────────
        st.markdown("###  LLM Provider")
        llm_provider = st.selectbox(
            "Choose your AI model:",
            options=["OpenAI GPT-4o", "Google Gemini 1.5 Pro"],
            index=0,
            help="Switch between OpenAI and Gemini anytime.",
        )

        # If provider changed → force chain rebuild
        if llm_provider != ss.get("llm_provider"):
            ss.llm_provider = llm_provider
            ss.rag_chain    = None
            if ss.get("llm_provider"):
                st.info(f"Switched to {llm_provider} — chain will rebuild on next query.")

        st.divider()

        # ── File Uploader ────────────────────────────────────
        valid_files = render_uploader()

        if valid_files:
            # Only process files not already indexed
            new_files = [
                f for f in valid_files
                if f.name not in ss.get("uploaded_filenames", [])
            ]

            if new_files:
                with st.spinner(f" Indexing {len(new_files)} file(s)..."):
                    try:
                        # ── Full ingestion pipeline ──────────
                        # Step 1: Load raw text
                        docs   = load_documents(new_files)

                        if docs:
                            # Step 2: Chunk
                            chunks = chunk_documents(docs)

                            # Step 3: Embed + store
                            if ss.vectorstore is None:
                                ss.vectorstore = build_vectorstore(chunks)
                            else:
                                ss.vectorstore = add_to_vectorstore(
                                    ss.vectorstore, chunks
                                )

                            # Step 4: Update stats
                            stats = get_document_stats(chunks)
                            ss.total_chunks += stats.get("total_chunks", 0)

                            # Step 5: Track filenames
                            for f in new_files:
                                ss.uploaded_filenames.append(f.name)

                            # Step 6: Force chain rebuild
                            ss.rag_chain = None

                            st.success(
                                f" Indexed {len(new_files)} file(s) "
                                f"→ {len(chunks)} chunks"
                            )
                            logger.info(
                                f"Indexed {len(new_files)} files, "
                                f"{len(chunks)} chunks"
                            )

                        else:
                            st.error("No text could be extracted from the files.")

                    except Exception as e:
                        st.error(f"Error during indexing: {str(e)}")
                        logger.error(f"Indexing error: {e}")

        st.divider()

        # ── Indexed Files Display ────────────────────────────
        if ss.get("uploaded_filenames"):
            st.markdown("###  Indexed Documents")

            for fname in ss.uploaded_filenames:
                st.markdown(f" `{fname}`")

            # Metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Files", len(ss.uploaded_filenames))
            with col2:
                st.metric("Chunks", ss.total_chunks)

            st.divider()

        # ── Chunking Settings Info ───────────────────────────
        with st.expander("🔧 Pipeline Settings"):
            st.markdown(f"""
            | Setting | Value |
            |---------|-------|
            | Chunk Size | `{config.CHUNK_SIZE}` chars |
            | Overlap | `{config.CHUNK_OVERLAP}` chars |
            | Top-K | `{config.TOP_K_RESULTS}` chunks |
            | Embedding | `{config.EMBEDDING_MODEL}` |
            | Search Type | `MMR` |
            """)
            st.caption("Edit `config.py` to tune these.")

        st.divider()

        # ── Clear Button ─────────────────────────────────────
        if st.button(" Clear Everything", type="secondary"):
            clear_vectorstore()
            ss.chat_history       = []
            ss.vectorstore        = None
            ss.rag_chain          = None
            ss.uploaded_filenames = []
            ss.total_chunks       = 0
            st.success("Cleared all documents and chat history!")
            logger.info("All data cleared by user")
            st.rerun()

        # ── API Key Status ───────────────────────────────────
        st.divider()
        st.markdown("###  API Key Status")
        openai_ok = bool(
            config.OPENAI_API_KEY and
            config.OPENAI_API_KEY != "sk-your-openai-key-here"
        )
        google_ok = bool(
            config.GOOGLE_API_KEY and
            config.GOOGLE_API_KEY != "your-google-gemini-key-here"
        )
        st.markdown(f"OpenAI: {' Connected' if openai_ok else ' Not set'}")
        st.markdown(f"Gemini: {' Connected' if google_ok else ' Not set'}")

    return llm_provider
