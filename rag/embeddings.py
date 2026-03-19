# ============================================================
# rag/embeddings.py — Text → Vector Embeddings
# ============================================================
#
# 📖 CONCEPT: What are Embeddings?
# ──────────────────────────────────
# An embedding is a list of numbers (vector) representing the
# MEANING of text in high-dimensional space.
#
# Example (simplified to 3D):
#   "king"         → [0.9,  0.1,  0.8]
#   "queen"        → [0.9,  0.1,  0.75]  ← close to king!
#   "banana"       → [0.1,  0.9,  0.2]   ← far from both
#
# OpenAI text-embedding-3-small → 1536-dimensional vectors
#
# 📖 CONCEPT: Why Embeddings for RAG?
# ─────────────────────────────────────
# Keyword search:  "heart attack" won't match "cardiac arrest"
# Embedding search: both phrases have SIMILAR vectors → matched ✅
#
# Semantic search understands MEANING, not just words.
#
# 📖 CONCEPT: Cosine Similarity
# ───────────────────────────────
# How we measure if two vectors are "close":
#   Score = 1.0  → identical meaning
#   Score = 0.0  → completely unrelated
#   Score = -1.0 → opposite meaning
#
# At query time:
#   1. Embed the user's question → query vector
#   2. Compare query vector against all stored chunk vectors
#   3. Return top-K chunks with highest cosine similarity

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils.logger import get_logger
import config

logger = get_logger()


def get_embedding_model():
    """
    Returns Google Gemini embedding model.
    Using embedding-001 — free with Gemini API key.
    """
    if not config.GOOGLE_API_KEY:
        raise ValueError(
            "Google API key not found. "
            "Please set GOOGLE_API_KEY in your .env file."
        )

    logger.info("Loading Gemini embedding model")

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=config.GOOGLE_API_KEY,
    )