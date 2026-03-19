# ============================================================
# config.py — Central Configuration
# ============================================================
# Single source of truth for all settings.
# Every other file imports from here — never hardcode values
# anywhere else in the project.

import os
from dotenv import load_dotenv

# Load .env file into environment variables
load_dotenv()

# ── API Keys ─────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# ── LLM Settings ─────────────────────────────────────────────
OPENAI_MODEL     = "gpt-4o"
GEMINI_MODEL     = "gemini-1.5-pro"
LLM_TEMPERATURE  = 0.3
LLM_MAX_TOKENS   = 1024

# ── Embedding Settings ───────────────────────────────────────
EMBEDDING_MODEL  = "text-embedding-3-small"

# ── Chunking Settings ────────────────────────────────────────
CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200

# ── ChromaDB Settings ────────────────────────────────────────
CHROMA_PERSIST_DIR  = "./chroma_db"
CHROMA_COLLECTION   = "research_docs"

# ── Retrieval Settings ───────────────────────────────────────
TOP_K_RESULTS    = 5

# ── File Upload Settings ─────────────────────────────────────
ALLOWED_EXTENSIONS   = [".pdf", ".docx", ".txt"]
MAX_FILE_SIZE_MB     = 20

# ── UI Settings ──────────────────────────────────────────────
APP_TITLE        = "📚 AI Research Assistant"
APP_ICON         = "📚"
APP_SUBTITLE     = "Upload documents and chat with them using RAG + LLM"