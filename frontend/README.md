# Private Local Document Summarizer

A **100% private, fully local** AI document summarization platform. Your documents never leave your machine.

## 🔒 Privacy-First Features

- **Zero Cloud Uploads** - All processing happens locally on your hardware
- **Powered by Ollama** - Uses open-source models (Mistral, Llama, Phi, etc.)
- **Offline Capable** - Works without internet connection
- **No Data Collection** - No tracking, no logging, no external APIs
- **GDPR Compliant** - Data never leaves your machine automatically

## 🚀 Getting Started

### Prerequisites

1. **Install Ollama** - [Download from ollama.ai](https://ollama.ai)
2. **Pull a model**:
   ```bash
   ollama pull mistral
   # or
   ollama pull llama2
   ```

### Run the Frontend

```bash
npm install
npm run dev
# or
pnpm install
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser.

### Backend Setup

The backend connects to your local Ollama instance. You can either:

1. **Use the unified backend script** (recommended):
   ```bash
   python backend-unified.py
   ```
   This script automatically detects your OS and installs all dependencies.

2. **Or use the legacy backend**:
   ```bash
   cd ../ai-backend
   python Hindices.py
   ```

Make sure Ollama is running:
```bash
ollama serve
```

## 🛠️ Tech Stack

- **Frontend**: Next.js 15, TypeScript, TailwindCSS
- **Backend**: FastAPI + LangChain
- **AI Engine**: Ollama (local)
- **Models**: Mistral, Llama, Phi, Gemma (any Ollama model)
- **Vector Store**: ChromaDB (local)

## 📦 Features

- ✅ **Map-Reduce Summarization** - Handles long documents efficiently
- ✅ **RAG (Retrieval Augmented Generation)** - Chat with your documents
- ✅ **PDF Support** - Extract and process PDF documents
- ✅ **Citation Tracking** - Source attribution for answers
- ✅ **Multi-Model Support** - Switch between different Ollama models

## 🔐 Why Local?

| Cloud Solutions | Our Local Approach |
|----------------|-------------------|
| Documents uploaded to servers | Documents stay on your machine |
| Pay per API call | Free, unlimited processing |
| Requires internet | Works offline |
| Vendor lock-in | You control everything |
| Privacy concerns | 100% private |

## 📄 License

Open source - your data, your machine, your control.

## 🤝 Contributing

Contributions welcome! This is a privacy-focused project - no cloud dependencies allowed.
