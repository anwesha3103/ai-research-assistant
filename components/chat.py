import streamlit as st
from streamlit import session_state as ss

from rag.vectorstore import get_retriever
from rag.chain import get_llm, build_rag_chain, query_chain, format_sources
from utils.logger import get_logger

logger = get_logger()


def render_chat():
    if not ss.get("uploaded_filenames"):
        st.markdown("""
        ###  Welcome to AI Research Assistant!

        **Get started in 3 steps:**

        1.  Add your API keys — copy `.env.example` to `.env` and fill in your keys
        2.  Upload documents — use the sidebar to upload PDF, DOCX, or TXT files
        3.  Start chatting — ask anything about your documents!

        ---

        ####  Example questions:
        - *"Summarize the main findings of this paper"*
        - *"What methodology was used in the study?"*
        - *"List all key concepts mentioned in chapter 2"*
        - *"What are the limitations mentioned by the author?"*
        - *"Compare the approaches described in section 3 and 5"*
        """)
        return

    st.markdown("###  Chat")

    # ── Render chat history ───────────────────────────────────
    for msg in ss.get("chat_history", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander(" View Sources", expanded=False):
                    st.markdown(msg["sources"])

    # ── Quick start buttons ───────────────────────────────────
    if not ss.get("chat_history"):
        st.markdown("####  Quick Start:")
        col1, col2, col3 = st.columns(3)
        quick_prompts = [
            (" Summarize",      "Give me a comprehensive summary of all uploaded documents"),
            (" Key Concepts",   "What are the most important concepts and terms in these documents?"),
            (" Main Questions", "What are the main research questions or problems addressed?"),
        ]
        for col, (label, prompt) in zip([col1, col2, col3], quick_prompts):
            with col:
                if st.button(label, use_container_width=True):
                    ss.chat_history.append({"role": "user", "content": prompt})
                    st.rerun()

    # ── Chat input ────────────────────────────────────────────
    question = st.chat_input("Ask anything about your documents...")

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        ss.chat_history.append({"role": "user", "content": question})

        # ── Build chain if needed ─────────────────────────────
        if ss.get("rag_chain") is None:
            if ss.get("vectorstore") is None:
                st.error("No documents indexed yet. Please upload files first.")
                return
            try:
                with st.spinner(" Initialising AI chain..."):
                    llm          = get_llm(ss.llm_provider)
                    retriever    = get_retriever(ss.vectorstore)
                    ss.rag_chain = build_rag_chain(retriever, llm)
                    logger.info(f"Chain built with {ss.llm_provider}")
            except ValueError as e:
                st.error(f" {str(e)}")
                st.info(" Check your API keys in the `.env` file.")
                return

        # ── Query ─────────────────────────────────────────────
        with st.chat_message("assistant"):
            with st.spinner(" Thinking..."):
                try:
                    # Build history tuples from chat history
                    history = ss.get("chat_history", [])
                    history_tuples = []
                    for i in range(0, len(history) - 1, 2):
                        if (history[i]["role"] == "user" and
                                i + 1 < len(history) and
                                history[i + 1]["role"] == "assistant"):
                            history_tuples.append((
                                history[i]["content"],
                                history[i + 1]["content"]
                            ))

                    answer, source_docs = query_chain(
                        ss.rag_chain,
                        question,
                        history_tuples
                    )
                    sources_text = format_sources(source_docs)

                    st.markdown(answer)

                    if sources_text:
                        with st.expander("📎 View Sources", expanded=False):
                            st.markdown(sources_text)

                    ss.chat_history.append({
                        "role":    "assistant",
                        "content": answer,
                        "sources": sources_text,
                    })
                    logger.info("Response delivered successfully")

                except Exception as e:
                    st.error(f" Error: {str(e)}")
                    logger.error(f"Query error: {e}")
                    if "api key" in str(e).lower():
                        st.info(" Check your API key in the `.env` file.")
                    elif "quota" in str(e).lower() or "rate" in str(e).lower():
                        st.info("! API rate limit hit — wait a moment and retry.")
