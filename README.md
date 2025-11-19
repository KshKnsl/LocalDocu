# LocalDocu: Private Local AI Document Summarizer

A **100% private, fully local** AI-powered document summarization and Q&A platform. Your documents never leave your machine - everything runs locally using Ollama and open-source models.

![Project Banner](https://img.shields.io/badge/Status-Active-green) ![License](https://img.shields.io/badge/License-MIT-blue) ![Privacy](https://img.shields.io/badge/Privacy-Local--First-orange)

## 🔒 Privacy-First Design

- **Zero Cloud Dependencies** - All processing happens on your local hardware
- **No Data Collection** - No tracking, logging, or external API calls
- **Offline Capable** - Works without internet connection
- **GDPR Compliant** - Your data stays on your machine
- **Open Source** - Transparent and auditable codebase

## 🚀 Quick Start

### Prerequisites

1. **Install Ollama** - [Download from ollama.ai](https://ollama.ai)
2. **Pull Required Models**:
   ```bash
   ollama pull gemma3:1b
   ollama pull llava
   ```

### Option 1: Automated Setup (Recommended)

Download the unified backend zip from [GitHub Releases](https://github.com/KshKnsl/LocalDocu/releases):

- **Universal**: `backend-unified.zip` (works on Windows, macOS, and Linux)

Extract the zip file and run the executable inside - it will automatically:
- Detect your operating system
- Install Python dependencies
- Set up Ollama with the official install script
- Pull required models
- Start the backend server

### Option 2: Manual Setup

#### Backend Setup

```bash
# Navigate to backend directory
cd ai-backend

# Install Python dependencies
pip install -r requirements.txt

# Configure environment (optional)
cp .env.example .env
# Edit .env with your ngrok token if needed

# Start the backend
python Hindices.py
```

#### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
# or
pnpm install

# Start development server
npm run dev
# or
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Ollama        │
│   (Next.js)     │◄──►│   (FastAPI)     │◄──►│   (Local LLMs)  │
│                 │    │                 │    │                 │
│ • File Upload   │    │ • Document      │    │ • Text Models   │
│ • Chat Interface│    │   Processing    │    │   (Gemma3:1b)   │
│ • Progress UI   │    │ • RAG Pipeline  │    │ • Vision Models │
│ • Citation View │    │ • API Endpoints │    │   (LLaVA)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Vector Store  │
                       │   (ChromaDB)    │
                       │                 │w
                       │ • Summary Store │
                       │ • Detail Store  │
                       │ • Image Store   │
                       └─────────────────┘
```

## ✨ Features

### Core Capabilities
- **📄 PDF Processing** - Extract and analyze PDF documents
- **🖼️ Image Understanding** - Q&A with images using LLaVA
- **💬 Interactive Chat** - Ask questions about your documents
- **📚 Citation Tracking** - Source attribution for all answers
- **🔄 Real-time Progress** - Live updates during processing

### Advanced RAG Features
- **Hierarchical Retrieval** - Two-tier vector search (summary + detailed)
- **Semantic Chunking** - Intelligent document splitting
- **Re-ranking** - Improved retrieval quality with FlashRank
- **Structured Citations** - IEEE-style references with deduplication
- **Multi-Modal Support** - Text + image processing

### Privacy & Performance
- **Local Processing** - No cloud uploads or API calls
- **Model Flexibility** - Support for any Ollama model
- **Efficient Storage** - ChromaDB for fast vector operations
- **Streaming Responses** - Real-time answer generation

## 📁 Project Structure

```
LocalDocu/
├── ai-backend/              # FastAPI backend (legacy)
│   ├── Hindices.py         # Legacy backend application
│   ├── requirements.txt    # Python dependencies
│   └── README.md           # Backend documentation
├── frontend/                # Next.js frontend
│   ├── src/
│   │   ├── app/            # Next.js app router
│   │   ├── components/     # React components
│   │   └── lib/            # Utilities
│   ├── public/             # Static files & unified backend
│   │   ├── backend-unified.py  # Cross-platform backend script
│   │   └── ...
│   ├── package.json
│   └── README.md           # Frontend documentation
├── progress-service/        # Progress tracking microservice
├── proxy-backend/           # Proxy service for deployment
├── chroma_store/            # Vector database storage
│   ├── summary_store/      # Document-level summaries
│   └── detailed_store/     # Chunk-level details
├── image_store/             # Extracted images
├── requirements.txt         # Root dependencies
└── README.md               # This file
```

## 🛠️ Tech Stack

### Backend
- **FastAPI** - High-performance async web framework
- **LangChain** - LLM orchestration and RAG pipelines
- **ChromaDB** - Vector database for embeddings
- **Ollama** - Local LLM inference
- **PyMuPDF** - PDF processing
- **Sentence Transformers** - Text embeddings

### Frontend
- **Next.js 15** - React framework with app router
- **TypeScript** - Type-safe JavaScript
- **Tailwind CSS** - Utility-first styling
- **shadcn/ui** - Modern UI components

### AI/ML
- **Ollama Models** - Gemma3:1b, LLaVA, Mistral, etc.
- **HuggingFace Embeddings** - all-MiniLM-L6-v2
- **FlashRank** - Re-ranking for better retrieval

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the backend directory:

```bash
# Backend Configuration
OLLAMA_MODEL=gemma3:1b
OLLAMA_URL=http://localhost:11434

# Optional: For public access
NGROK_AUTHTOKEN=your_ngrok_token_here

# Storage Paths
PERSIST_BASE=./chroma_store
IMAGE_STORE=./image_store

# Progress Tracking
PROGRESS_SERVICE_URL=https://your-progress-service.vercel.app
```

### Model Configuration

The system uses two types of models:
- **Text Models**: For summarization and Q&A (default: gemma3:1b)
- **Vision Models**: For image understanding (default: llava)

You can change models by:
1. Pulling new models: `ollama pull <model_name>`
2. Updating `OLLAMA_MODEL` in `.env`
3. Restarting the backend

## 🚀 Deployment

### Local Development
```bash
# Backend
cd ai-backend && python Hindices.py

# Frontend (new terminal)
cd frontend && npm run dev
```

### Production Build

#### Frontend
```bash
cd frontend
npm run build
npm start
```

#### Backend Executables
The project includes automated GitHub Actions to build a unified executable that works on all platforms:
- Universal executable: `backend-unified.zip`

Download from [Releases](https://github.com/KshKnsl/LocalDocu/releases)

### Docker (Future)
Docker support planned for easier deployment.

## 📊 API Reference

### Core Endpoints

#### `POST /process`
Upload and process documents.

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
  "chunkCount": 42
}
```

#### `POST /generate`
Query documents with RAG.

**Request:**
```json
{
  "prompt": "What are the key findings?",
  "documentIds": ["doc_abc123"],
  "model": "gemma3:1b"
}
```

**Response:**
```json
{
  "response": "The key findings are... [1]",
  "citations": [
    {
      "documentId": "document.pdf",
      "page": 5,
      "snippet": "...",
      "source": "Research Paper"
    }
  ]
}
```

## 🔍 Troubleshooting

### Common Issues

#### Ollama Not Starting
```bash
# Check if Ollama is installed
ollama --version

# Start Ollama service
ollama serve

# Check if models are available
ollama list
```

#### Backend Won't Start
- Ensure Python 3.8+ is installed
- Check if port 8000 is available
- Verify Ollama is running on port 11434

#### Frontend Connection Issues
- Ensure backend is running on localhost:8000
- Check CORS settings if deploying separately

#### Memory Issues
- Use smaller models (gemma3:1b instead of larger models)
- Process fewer documents simultaneously
- Close other applications to free RAM

### Performance Tips
- Use SSD storage for faster vector operations
- Keep ChromaDB stores on fast drives
- Use smaller embedding models for better speed

## 🤝 Contributing

We welcome contributions! This project focuses on privacy and local processing.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally with Ollama
5. Submit a pull request

### Guidelines
- Maintain local-first philosophy
- No cloud dependencies
- Comprehensive testing
- Clear documentation

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ollama** - For making local LLMs accessible
- **LangChain** - For RAG framework
- **ChromaDB** - For vector storage
- **SurfSense** - Inspiration for hierarchical RAG

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/KshKnsl/LocalDocu/issues)
- **Discussions**: [GitHub Discussions](https://github.com/KshKnsl/LocalDocu/discussions)
- **Documentation**: See individual README files in subdirectories

---

**Built with ❤️ for privacy and local AI processing**