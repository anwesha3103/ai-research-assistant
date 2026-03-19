# ============================================================
# rag/embeddings.py — Text → Vector Embeddings
# ============================================================


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
