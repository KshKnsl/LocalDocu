# LocalDocu: Complete Project Report

---

## Table of Contents
- [ai-backend/](#ai-backend)
  - [Hindices.py](#hindicespy-detailed-architecture)
  - README.md
  - requirements.txt
- [frontend/](#frontend)
  - src/
  - public/
  - package.json, config files
- [progress-service/](#progress-service)
- [proxy-backend/](#proxy-backend)
- [chroma_store/](#chroma_store)
- [image_store/](#image_store)
- [requirements.txt (root)](#requirements-txt-root)

---

## ai-backend/

### Hindices.py (Detailed Architecture)

**Purpose:**
The core backend for hierarchical Retrieval-Augmented Generation (RAG) with multi-modal (text + image) support. Implements document ingestion, chunking, semantic search, advanced summarization, and structured citation generation. Designed for research and production, with a focus on explainability, modularity, and extensibility.

**Key Features:**
- **Hierarchical RAG:** Two-level vector store (summary + detailed chunks) for efficient, context-rich retrieval.
- **Semantic Chunking:** Uses LangChain's SemanticChunker for intelligent document splitting.
- **Image Understanding:** Integrates LLaVA for local vision-language inference.
- **Structured Citations:** Generates IEEE-style citations, deduplicates references, and maintains traceability.
- **Streaming Summarization:** Supports progressive, chunk-level summarization with real-time progress updates.
- **Progress Tracking:** Posts progress to a separate service for robust frontend feedback.
- **Local-First LLMs:** Uses Ollama for all LLM inference (Mistral, LLaVA, etc.).
- **Robust Error Handling:** Always returns JSON responses, never relies on HTTP status codes for progress/errors.

**Major Components:**
- **Configuration & Environment:** Loads API keys, model names, and persistent storage paths from `.env` or Colab secrets. Ensures all required directories exist.
- **Document Ingestion:** Handles PDF and image uploads, splits PDFs into semantically meaningful chunks, extracts and summarizes images, and stores all data in ChromaDB.
- **Summarization:**
  - Chunk-level: Uses a small, fast model (e.g., Mistral) for initial summaries.
  - Synthesis: Optionally uses a larger model for final document synthesis.
- **RAG Query:**
  - Retrieves relevant summaries and detailed chunks using semantic search and re-ranking.
  - Builds advanced prompts for LLMs, enforces structured output, and deduplicates citations.
- **Image Q&A:**
  - Always uses LLaVA for vision tasks, regardless of user model selection.
- **API Endpoints:**
  - `/process`: Ingests and processes documents/images, always returns JSON with progress.
  - `/generate`: Answers user queries with hierarchical retrieval and structured citations.
  - `/get_chunks`: Returns all chunks for a document.
  - `/pull`: Pulls models via Ollama.
  - `/image/{image_id}` and `/image_bytes/{image_id}`: Serve images by ID. 
- **Startup Logic:**
  - Ensures Ollama is running and pulls both `mistral` and `llava` models by default.
  - Optionally configures ngrok for public access.

**Research/Engineering Rationale:**
- **Hierarchical RAG** improves retrieval quality and efficiency by combining coarse (summary) and fine (detailed chunk) search.
- **Semantic chunking** ensures contextually meaningful splits, improving downstream LLM performance.
- **Structured citations** are critical for research transparency and reproducibility.
- **Local-first, multi-modal LLMs** (Ollama + LLaVA) ensure privacy, speed, and cost-effectiveness.
- **Progress tracking** decouples backend errors from HTTP status, making the system robust to network issues and frontend timeouts.

---

### README.md
- Setup, install, and usage instructions for the backend. Lists all features, dependencies, and API endpoints.

### requirements.txt
- Python dependencies for the backend (FastAPI, LangChain, ChromaDB, etc.).

---

## frontend/
- **src/app/**: Next.js app routes, API handlers, and page components.
- **src/components/**: All UI components (chat, sidebar, input, file preview, etc.).
- **src/lib/**: Client-side utilities for API calls, chat storage, model management, etc.
- **public/**: Static files and backend scripts for different OSes.
- **package.json, config files**: Frontend dependencies and build configuration.

---

## progress-service/
- Node.js microservice for tracking document processing progress. Receives updates from the backend and exposes them to the frontend for real-time feedback.

---

## proxy-backend/
- Lightweight Python backend for proxying requests (e.g., for deployment or CORS).

---

## chroma_store/
- Persistent vector store directories for ChromaDB (summary_store, detailed_store).

---

## image_store/
- Stores all extracted and uploaded images for document Q&A and summarization.

---

## requirements.txt (root)
- Top-level dependency list for the entire project (may include both backend and frontend requirements).

---

# End of Report
