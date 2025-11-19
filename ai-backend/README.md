# AI Backend - Hierarchical RAG with SurfSense

A production-ready FastAPI backend implementing hierarchical RAG (Retrieval-Augmented Generation) with advanced features inspired by SurfSense.

## Features

- **Hierarchical Vector Stores**: Two-tier RAG architecture (summary + detailed chunks)
- **Semantic Chunking**: Intelligent document splitting using LangChain's SemanticChunker
- **Advanced Re-ranking**: FlashrankRerank for improved retrieval quality
- **Structured Citations**: IEEE-style citations with reference deduplication
- **Streaming Summarization**: Progressive document summarization with chunk-level updates
- **Multi-Modal Support**: Text documents (PDF) + image understanding (via LLaVA)
- **Local LLMs**: Supports Ollama for local inference

## Prerequisites

- Python 3.9+
- Ollama (for local LLM inference) - [Install](https://ollama.com)
- Ngrok (for public tunneling) - [Install](https://ngrok.com/download)

## Quick Start

### 1. Clone and Setup

```bash
cd ai-backend
```

### 2. Install Dependencies

```bash
pip install fastapi uvicorn pyngrok requests boto3 python-multipart aiofiles \
  langchain langchain-community chromadb sentence-transformers PyMuPDF \
  langchain-huggingface langchain-chroma \
  langchain-ollama langchain-experimental flashrank-retriever pydantic
```

### 3. Configure Environment

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```bash
NGROK_AUTHTOKEN=your_actual_ngrok_token
```

### 4. Start Ollama & Pull Models

```bash
# Start Ollama service
ollama serve

# In another terminal, pull required models
ollama pull mistral
ollama pull llava
```

### 5. Run the Server

#### Option A: Standalone Python (Recommended for Local Development)
```bash
python run_server.py
```

The runner script will:
- Check if `.env` file exists
- Verify Ollama service is running
- Start the FastAPI server on port 8000

#### Option B: Direct Execution (Colab/Jupyter)
```python
%run Hindices.py
```

Note: For Colab, uncomment the install commands at the top of `Hindices.py`.

## API Endpoints

### Core Endpoints

#### `POST /process`
Upload and process documents (PDF/images) into the hierarchical vector stores.

**Request:**
```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "documentId": "doc_abc123",
  "status": "embeddings_created",
  "chunkCount": 42,
  "isImage": false
}
```

#### `POST /generate`
Query the RAG system with hierarchical retrieval and structured citations.

**Request:**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the key findings?",
    "model": "mistral",
    "documentIds": ["doc_abc123"]
  }'
```

**Response:**
```json
{
  "response": "The key findings are... [1][2]",
  "citations": [
    {
      "id": "1",
      "title": "Research Paper (Page 5, Chunk 2)",
      "source": "document.pdf"
    }
  ]
}
```

#### `POST /summarize_by_id`
Stream progressive summarization of a document.

**Request:**
```bash
curl -X POST "http://localhost:8000/summarize_by_id" \
  -H "Content-Type: application/json" \
  -d '{
    "documentId": "doc_abc123",
    "model_name": "mistral"
  }'
```

**Response (NDJSON stream):**
```
{"type": "chunk", "index": 0, "total": 10, "summary": "..."}
{"type": "chunk", "index": 1, "total": 10, "summary": "..."}
{"type": "status", "message": "Generating final summary..."}
{"type": "final", "summary": "...", "chunkCount": 10}
```

#### `POST /pull`
Pull/download an Ollama model.

**Request:**
```bash
curl -X POST "http://localhost:8000/pull" \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3"}'
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│          Document Ingestion Pipeline            │
├─────────────────────────────────────────────────┤
│  PDF → PyMuPDF → SemanticChunker → Embeddings  │
│         ↓                    ↓                   │
│   Summary Store      Detailed Store             │
│   (doc-level)        (chunk-level)              │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│         Hierarchical RAG Query Flow             │
├─────────────────────────────────────────────────┤
│  1. Query → Summary Store (re-ranked)           │
│       ↓                                          │
│  2. Top docs → Detailed Store (re-ranked)       │
│       ↓                                          │
│  3. Top chunks → SurfSense Prompt → LLM         │
│       ↓                                          │
│  4. Structured Output + Citations                │
└─────────────────────────────────────────────────┘
```

## Configuration

All settings can be configured via environment variables (see `.env.example`):

- **API Keys**: `NGROK_AUTHTOKEN`
- **Models**: `OLLAMA_MODEL`, `EMBEDDINGS_MODEL`
- **Storage**: `PERSIST_BASE`, `IMAGE_STORE`
- **RAG**: `RERANKER_TOP_N`, `MAX_CITATIONS`
- **Server**: `SERVER_HOST`, `SERVER_PORT`, `LOG_LEVEL`

## Development

### Project Structure

```
ai-backend/
├── Hindices.py          # Main application
├── .env                 # Your secrets (not committed)
├── .env.example         # Template for .env
├── .gitignore           # Git ignore rules
├── requirements.txt     # Python dependencies (to be created)
└── README.md           # This file
```

### Adding Dependencies

Update the pip install command in this README and create a `requirements.txt`:

```bash
pip freeze > requirements.txt
```

## Troubleshooting

### Ollama not starting
- Ensure Ollama is installed: `ollama --version`
- Check if port 11434 is available: `lsof -i :11434` (macOS/Linux) or `netstat -ano | findstr :11434` (Windows)

### Out of memory errors
- Reduce chunk size in `SemanticChunker`
- Use smaller embedding models
- Process fewer documents simultaneously

### Structured output fails
- Ensure you're using a compatible model (mistral, llama3)
- Check model supports JSON mode: `ollama show <model>`

## Credits

Inspired by [SurfSense](https://github.com/MODSetter/SurfSense) - an open-source alternative to NotebookLM.

## License

[Your License Here]
