# ============================================================
# rag/chain.py — RAG Chain + LLM Switching
# ============================================================
#
# 📖 CONCEPT: What is a RAG Chain?
# ──────────────────────────────────
# Without RAG:
#   User: "What does my PDF say about transformers?"
#   LLM:  "I don't have access to your PDF." ❌
#
# With RAG:
#   1. Embed the user's question
#   2. Retrieve top-K similar chunks from ChromaDB  ← Retrieval
#   3. Build prompt: "Context: {chunks} Question: {question}"
#   4. LLM answers using ONLY the provided context  ← Augmented Generation
#
# 📖 CONCEPT: Why does RAG reduce hallucination?
# ────────────────────────────────────────────────
# Without RAG → LLM answers from training memory → can confabulate
# With RAG    → LLM is GROUNDED to retrieved text → must stay faithful
# We enforce this via the system prompt: "only use provided context"
#
# 📖 CONCEPT: Conversational Memory
# ───────────────────────────────────
# LLMs are stateless — they don't remember previous messages.
# ConversationBufferWindowMemory fixes this by appending the
# last K exchanges to every new prompt.
#
# Without memory:
#   User: "Who wrote this paper?"     → LLM answers ✅
#   User: "What else did they write?" → LLM: "Who?" ❌
#
# With memory (k=10):
#   Full chat history injected into every prompt → follow-ups work ✅
#
# 📖 CONCEPT: LLM Abstraction
# ─────────────────────────────
# Both OpenAI and Gemini implement LangChain's BaseChatModel.
# Our chain only talks to BaseChatModel → swapping providers
# requires zero changes to chain logic. Pure abstraction. ✅

from typing import Tuple, List

from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.logger import get_logger
import config

logger = get_logger()

SYSTEM_PROMPT = """You are an expert AI Research Assistant helping users understand their uploaded documents deeply and accurately.

STRICT RULES:
1. Answer ONLY using the context provided below. Do not use outside knowledge.
2. If the context doesn't contain enough information, say exactly: "I couldn't find enough information in the uploaded documents to answer this confidently."
3. Always cite sources at the end using: [Source: filename, page X]
4. Be thorough but concise. Use bullet points for complex answers.
5. For summaries, use clear headings and structured key points.
6. Never make up facts, statistics, or quotes not present in the context.

CONTEXT FROM DOCUMENTS:
{context}
"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def get_llm(provider: str):
    logger.info(f"Initialising LLM: {provider}")

    if provider == "OpenAI GPT-4o":
        if not config.OPENAI_API_KEY:
            raise ValueError("OpenAI API key not set in .env file.")
        return ChatOpenAI(
            model=config.OPENAI_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            openai_api_key=config.OPENAI_API_KEY,
            streaming=True,
        )

    elif provider == "Google Gemini 1.5 Pro":
        if not config.GOOGLE_API_KEY:
            raise ValueError("Google API key not set in .env file.")
        return ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_output_tokens=config.LLM_MAX_TOKENS,
            google_api_key=config.GOOGLE_API_KEY,
            convert_system_message_to_human=True,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


def build_rag_chain(retriever, llm):
    logger.info("Building RAG chain...")

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        verbose=False,
    )

    logger.info("✅ RAG chain ready")
    return chain


def query_chain(
    chain,
    question: str,
    chat_history: list = []
) -> Tuple[str, List[Document]]:
    history_tuples = []
    messages = chat_history or []
    for i in range(0, len(messages) - 1, 2):
        if messages[i]["role"] == "user" and i+1 < len(messages) and messages[i+1]["role"] == "assistant":
            history_tuples.append((
                messages[i]["content"],
                messages[i+1]["content"]
            ))

    result = chain({
        "question": question,
        "chat_history": history_tuples
    })

    answer      = result.get("answer", "No answer generated.")
    source_docs = result.get("source_documents", [])

    logger.info(f"Answer generated using {len(source_docs)} source chunks")
    return answer, source_docs


def format_sources(source_docs: List[Document]) -> str:
    if not source_docs:
        return ""

    seen  = set()
    lines = ["📎 **Sources Used:**"]

    for doc in source_docs:
        source = doc.metadata.get("source", "Unknown")
        page   = doc.metadata.get("page", "?")
        key    = f"{source}-{page}"

        if key not in seen:
            seen.add(key)
            lines.append(f"  • `{source}` — Page {page}")

    return "\n".join(lines)