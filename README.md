# AI Research Assistant
### RAG-based Document Question Answering System

Upload your documents. Ask anything. Get grounded, cited answers — powered by a full Retrieval-Augmented Generation pipeline.

---

## Overview

AI Research Assistant is a production-grade application that lets you have a conversation with your documents. Built on a modular RAG architecture, it processes PDFs, Word documents, and text files — chunking, embedding, and indexing them into a vector database — then answers your questions using only the content of your uploaded files, with source citations to back every response.

No hallucinations. No outside knowledge. Just your documents, made queryable.

---

## Architecture

```
Document Upload
      │
      ▼
 [Loader]              Extract text from PDF / DOCX / TXT
      │
      ▼
 [Chunker]             RecursiveCharacterTextSplitter
                       chunk_size=1000, overlap=200
      │
      ▼
 [Embeddings]          Gemini embedding-001 → 768-dim vectors
      │
      ▼
 [ChromaDB]            Persistent local vector store
      │
      ▼
 [Retriever]           MMR search — top-5 diverse, relevant chunks
      │
      ▼
 [RAG Chain]           Context + question → LLM prompt
      │
      ▼
 [LLM]                 Gemini 1.5 Pro / GPT-4o (switchable)
      │
      ▼
 Answer + Citations
```

---

## Features

- **Semantic search** — finds relevant content by meaning, not keywords
- **MMR retrieval** — Maximal Marginal Relevance ensures diverse, non-redundant context
- **Switchable LLMs** — toggle between Gemini 1.5 Pro and GPT-4o from the UI
- **Source citations** — every answer references the exact document and page
- **Conversational memory** — follow-up questions work naturally across turns
- **Hallucination reduction** — LLM is strictly grounded to retrieved document context
- **Multi-format support** — PDF, DOCX, and TXT ingestion
- **Persistent vector store** — ChromaDB persists to disk, no re-indexing on restart
- **File validation** — size and format checks before ingestion

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| Embeddings | Gemini `embedding-001` |
| Vector Store | ChromaDB |
| LLM (primary) | Google Gemini 1.5 Pro |
| LLM (alternate) | OpenAI GPT-4o |
| Document Loading | LangChain Community Loaders |
| Logging | Loguru |
| Testing | Pytest |

---

## Project Structure

```
ai-research-assistant/
│
├── app.py                      # Streamlit entry point
├── config.py                   # Centralised configuration
├── requirements.txt
├── .env.example                # API key template
│
├── rag/                        # Core RAG pipeline
│   ├── loader.py               # Document ingestion + chunking
│   ├── embeddings.py           # Embedding model interface
│   ├── vectorstore.py          # ChromaDB operations
│   └── chain.py                # RAG chain + LLM abstraction
│
├── components/                 # Streamlit UI components
│   ├── sidebar.py              # Settings + file upload
│   ├── chat.py                 # Chat interface + query handling
│   └── uploader.py             # File validation + upload widget
│
├── utils/
│   ├── logger.py               # Loguru logging setup
│   └── file_validator.py       # File type + size validation
│
├── tests/                      # Unit tests
│   ├── test_loader.py
│   ├── test_vectorstore.py
│   └── test_chain.py
│
└── docs/
    ├── architecture.md
    └── usage.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/anwesha3103/ai-research-assistant.git
cd ai-research-assistant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp .env.example .env
```

Open `.env` and add your keys:

```
OPENAI_API_KEY=your-openai-key        # optional
GOOGLE_API_KEY=your-gemini-key        # required for default setup
```

Get a free Gemini key at [aistudio.google.com](https://aistudio.google.com/app/apikey)

### 4. Run the app

```bash
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## Configuration

All tuneable parameters live in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 200 | Overlap between adjacent chunks |
| `TOP_K_RESULTS` | 5 | Chunks retrieved per query |
| `LLM_TEMPERATURE` | 0.3 | Response determinism (0 = strict) |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `MAX_FILE_SIZE_MB` | 20 | Upload size limit |

---

## Example Usage

Upload a research paper and ask:

- *"What is the main contribution of this paper?"*
- *"Summarise the methodology used in section 3"*
- *"What datasets were used for evaluation?"*
- *"What are the stated limitations of this approach?"*
- *"How does this compare to prior work?"*

Each response includes source citations with filename and page number.

---

## Troubleshooting

**Invalid API key** — ensure your `.env` file has no extra spaces around the `=` sign.

**ChromaDB errors** — delete the `chroma_db/` folder and re-upload your documents.

**Slow first upload** — expected. The first run embeds all chunks via the Gemini API. Subsequent runs load the index from disk instantly.

**Module not found** — run `pip install -r requirements.txt` again.

---

## License

MIT