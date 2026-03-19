# ============================================================
# rag/vectorstore.py — ChromaDB Vector Store Interface
# ============================================================

from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from rag.embeddings import get_embedding_model
from utils.logger import get_logger
import config

logger = get_logger()


def build_vectorstore(chunks: List[Document]) -> Chroma:
    """
    Embed all chunks and store them in ChromaDB.
    This is the INDEXING phase — runs once per document upload.

    Steps happening inside:
      1. Calls OpenAI API to embed every chunk (network call)
      2. Stores (text + embedding + metadata) in ChromaDB
      3. Persists everything to disk at CHROMA_PERSIST_DIR
    """
    logger.info(f"Building vector store with {len(chunks)} chunks...")

    embedding_model = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=config.CHROMA_PERSIST_DIR,
    )

    logger.info(f" Vector store built with {len(chunks)} vectors")
    return vectorstore


def load_vectorstore() -> Optional[Chroma]:
    """
    Load an existing ChromaDB collection from disk.
    Returns None if no collection exists yet.

    Called on app startup so we don't re-embed on every refresh.
    """
    try:
        embedding_model = get_embedding_model()

        vectorstore = Chroma(
            collection_name=config.CHROMA_COLLECTION,
            embedding_function=embedding_model,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )

        count = vectorstore._collection.count()

        if count == 0:
            logger.info("No existing vector store found")
            return None

        logger.info(f"Loaded existing vector store ({count} vectors)")
        return vectorstore

    except Exception as e:
        logger.warning(f"Could not load vector store: {e}")
        return None


def add_to_vectorstore(
    existing_store: Chroma,
    new_chunks: List[Document]
) -> Chroma:
    """
    Add new chunks to an already existing vector store.
    Used when user uploads additional files mid-session.
    """
    existing_store.add_documents(new_chunks)
    logger.info(f"Added {len(new_chunks)} new chunks to vector store")
    return existing_store


def clear_vectorstore() -> None:
    """
    Delete the persisted ChromaDB collection from disk.
    Called when user clicks 'Clear All Documents' in the UI.
    """
    import shutil
    import os

    if os.path.exists(config.CHROMA_PERSIST_DIR):
        shutil.rmtree(config.CHROMA_PERSIST_DIR)
        logger.info("Vector store cleared")


def get_retriever(vectorstore: Chroma):
    """
    Convert vector store into a LangChain Retriever.

    A Retriever is LangChain's standard interface for fetching
    relevant documents — abstracts away the underlying DB so
    the chain doesn't care if it's ChromaDB, FAISS, or Pinecone.

    search_type options:
      "similarity" → plain cosine similarity
      "mmr"        → Maximal Marginal Relevance (diverse + relevant) 
    """
    logger.info(f"Creating retriever (top_k={config.TOP_K_RESULTS})")

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": config.TOP_K_RESULTS,
            "fetch_k": 20,      # MMR candidate pool — pick best 5 from 20
        }
    )
