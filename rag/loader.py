# ============================================================
# rag/loader.py — Document Ingestion & Chunking
# ============================================================


import os
import tempfile
from typing import List

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

from utils.logger import get_logger
import config

logger = get_logger()


def load_documents(uploaded_files) -> List[Document]:
    """
    Accept Streamlit UploadedFile objects, save temporarily,
    load with the right LangChain loader, return LangChain Documents.

    Each Document has:
      .page_content  → extracted text
      .metadata      → filename, page number, file type
    """
    all_docs: List[Document] = []

    for uploaded_file in uploaded_files:

        # ── Save to temp file ────────────────────────────────
        # LangChain loaders need a real file path, not a file object
        suffix = os.path.splitext(uploaded_file.name)[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            # ── Pick loader by file type ─────────────────────
            if suffix == ".pdf":
                # Splits by page automatically, adds page number to metadata
                loader = PyPDFLoader(tmp_path)

            elif suffix in (".docx", ".doc"):
                # Extracts all text from Word documents
                loader = Docx2txtLoader(tmp_path)

            elif suffix == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")

            else:
                logger.warning(f"Unsupported file type: {suffix} — skipping")
                continue

            # ── Load & tag with source filename ──────────────
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"]    = uploaded_file.name
                doc.metadata["file_type"] = suffix

            all_docs.extend(docs)
            logger.info(f"Loaded {len(docs)} pages from {uploaded_file.name}")

        except Exception as e:
            logger.error(f"Error loading {uploaded_file.name}: {e}")

        finally:
            # Always clean up temp file
            os.unlink(tmp_path)

    logger.info(f"Total documents loaded: {len(all_docs)}")
    return all_docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller overlapping chunks using
    RecursiveCharacterTextSplitter — LangChain's smartest splitter.

     HOW RecursiveCharacterTextSplitter WORKS:
    ─────────────────────────────────────────────
    It tries separators in ORDER, falling back to the next if
    the chunk is still too large:

      1. "\n\n"  → paragraph break   (best, preserves meaning)
      2. "\n"    → line break
      3. ". "    → sentence end
      4. " "     → word boundary
      5. ""      → hard character split (last resort)

    This is smarter than splitting every N characters blindly
    because it respects natural text boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    logger.info(
        f"{len(documents)} pages → {len(chunks)} chunks "
        f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})"
    )

    return chunks


def get_document_stats(chunks: List[Document]) -> dict:
    """
    Return useful stats about the chunked documents.
    Displayed in the Streamlit sidebar.
    """
    if not chunks:
        return {}

    sources   = list(set(c.metadata.get("source", "unknown") for c in chunks))
    avg_len   = sum(len(c.page_content) for c in chunks) // len(chunks)

    return {
        "total_chunks":  len(chunks),
        "total_sources": len(sources),
        "sources":       sources,
        "avg_chunk_len": avg_len,
    }
