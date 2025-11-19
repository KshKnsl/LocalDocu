#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import requests
import platform
import tempfile
import shutil
import psutil
import json

def check_system_requirements():
    print("Checking system requirements...")

    mac_ver = platform.mac_ver()[0]
    if not mac_ver:
        print("Unable to determine macOS version")
        return False

    major, minor = map(int, mac_ver.split('.')[:2])
    if major < 10 or (major == 10 and minor < 15):
        print(f"macOS {mac_ver} is not supported. Minimum required: macOS 10.15 (Catalina)")
        return False
    print(f"macOS version: {mac_ver}")

    ram_gb = psutil.virtual_memory().total / (1024**3)
    if ram_gb < 8:
        print(f"Insufficient RAM: {ram_gb:.1f}GB. Minimum required: 8GB")
        return False
    print(f"RAM: {ram_gb:.1f}GB")

    disk_free_gb = psutil.disk_usage('/').free / (1024**3)
    if disk_free_gb < 10:
        print(f"Insufficient disk space: {disk_free_gb:.1f}GB free. Minimum required: 10GB")
        return False
    print(f"Free disk space: {disk_free_gb:.1f}GB")

    print("System requirements met")
    return True

def run_command(cmd, shell=True, check=True, timeout=300):
    try:
        result = subprocess.run(cmd, shell=shell, check=check, capture_output=True, text=True, timeout=timeout)
        return True, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def run_sudo_command(cmd, password=None, timeout=300):
    if password:
        full_cmd = f"echo '{password}' | sudo -S {cmd}"
    else:
        full_cmd = f"sudo {cmd}"

    try:
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return True, result.stdout + result.stderr
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)

def install_system_dependencies():
    print("Installing system dependencies...")

    success, output = run_command("which brew")
    if not success:
        print("Homebrew is required. Installing Homebrew...")
        install_cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        success, output = run_command(install_cmd)
        if not success:
            print(f"Failed to install Homebrew: {output}")
            return False
        print("Homebrew installed successfully")

    print("Updating Homebrew...")
    success, output = run_command("brew update")
    if not success:
        print(f"Warning: Failed to update Homebrew: {output}")

    success, output = run_command("which python3")
    if not success:
        print("Installing Python 3...")
        success, output = run_command("brew install python@3.11")
        if not success:
            print(f"Failed to install Python: {output}")
            return False
        print("Python installed successfully")

    print("System dependencies installed")
    return True

def install_python_packages():
    print("Installing Python dependencies...")
    packages = [
        "fastapi", "uvicorn", "pyngrok", "requests", "boto3", "python-multipart",
        "aiofiles", "langchain", "langchain-community", "chromadb", "sentence-transformers",
        "PyMuPDF", "langchain-huggingface", "langchain-chroma",
        "langchain-ollama", "langchain-experimental", "flashrank", "pydantic", "python-dotenv",
        "psutil"
    ]
    pip_cmd = "pip3" if shutil.which("pip3") else "python3 -m pip"
    cmd = f"{pip_cmd} install --upgrade pip"
    success, output = run_command(cmd)
    if not success:
        print(f"Warning: Failed to upgrade pip: {output}")
    cmd = f"{pip_cmd} install {' '.join(packages)}"
    success, output = run_command(cmd, timeout=600)
    if success:
        print("Python packages installed successfully")
        return True
    else:
        print(f"Failed to install Python packages: {output}")
        return False

def is_ollama_installed():
    success, output = run_command("which ollama")
    return success

def install_ollama():
    """Install Ollama using Homebrew."""
    print("Installing Ollama...")

    if is_ollama_installed():
        print("Ollama is already installed")
        return True

    success, output = run_command("brew install ollama")
    if success:
        print("Ollama installed successfully")
        return True
    else:
        print(f"Failed to install Ollama: {output}")
        return False

def wait_for_ollama_service(timeout=60):
    print("Waiting for Ollama service to start...")
    for i in range(timeout):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                print("Ollama service is running")
                return True
        except:
            pass
        time.sleep(1)
    print("Ollama service failed to start within timeout")
    return False

def pull_models():
    """Pull required Ollama models."""
    print("Pulling Ollama models (gemma3:1b and llava)...")
    models = ["gemma3:1b", "llava"]

    for model in models:
        print(f"Pulling {model}...")
        success, output = run_command(f"ollama pull {model}", timeout=600)  # 10 minute timeout
        if success:
            print(f"{model} pulled successfully")
        else:
            print(f"Failed to pull {model}: {output}")
            return False
    return True



def create_env_file():
    env_content = """# Backend Configuration
OLLAMA_MODEL=gemma3:1b
OLLAMA_URL=http://localhost:11434



"""
    with open('.env', 'w') as f:
        f.write(env_content)
    print("Created .env file with default configuration")

def main():
    print("Starting complete local backend setup for macOS...")
    print("=" * 60)
    if sys.version_info < (3, 8):
        print("Python 3.8 or higher is required")
        sys.exit(1)
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    if not check_system_requirements():
        sys.exit(1)
    if not install_system_dependencies():
        sys.exit(1)
    if not install_python_packages():
        sys.exit(1)
    if not install_ollama():
        sys.exit(1)
    print("Starting Ollama service...")
    success, output = run_command("ollama serve", check=False)
    if not success:
        print(f"Failed to start Ollama service: {output}")
        sys.exit(1)
    if not wait_for_ollama_service():
        sys.exit(1)
    if not pull_models():
        sys.exit(1)
    create_env_file()
    print("=" * 60)
    print("Setup complete! Starting backend server...")
    print("Backend will be available at: http://localhost:8000")
    print("Edit .env file to configure API keys or ngrok if needed")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    exec_backend_code()

def exec_backend_code():
    """Execute the backend code inline."""
    
    import os, signal, psutil, gc, time, sys, subprocess, threading, asyncio, json, base64
    from pathlib import Path
    from uuid import uuid4
    from typing import List, Dict, Any, Tuple
    from dotenv import load_dotenv
    
    load_dotenv()
    
    from fastapi import FastAPI, UploadFile, Form, Request, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, Response
    import uvicorn
    from pyngrok import ngrok
    
    from langchain.docstore.document import Document
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import FlashrankRerank
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_community.vectorstores.utils import filter_complex_metadata
    from langchain_experimental.text_splitter import SemanticChunker
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from langchain_ollama.llms import OllamaLLM
    from pydantic import BaseModel, Field
    import fitz
    
    try:
        from google.colab import userdata
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    
    print("Loading configuration...")
    
    if IN_COLAB:
        try:
            NGROK_AUTHTOKEN = "32eB7tLSQoICKJD4JSQuJ9lWea6_7U5ndjtQCVaWnPLEc4Mws"
            PROGRESS_SERVICE_URL = "https://minor-project-progress.vercel.app"
        except Exception:
            print("WARNING: Could not load from Colab secrets, falling back to environment variables.")
            NGROK_AUTHTOKEN = "32eB7tLSQoICKJD4JSQuJ9lWea6_7U5ndjtQCVaWnPLEc4Mws"
            PROGRESS_SERVICE_URL = "https://minor-project-progress.vercel.app"
    else:
        NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN", "32eB7tLSQoICKJD4JSQuJ9lWea6_7U5ndjtQCVaWnPLEc4Mws")
    
    if not NGROK_AUTHTOKEN or NGROK_AUTHTOKEN == "YOUR_NGROK_AUTHTOKEN":
        print("WARNING: NGROK_AUTHTOKEN not configured properly. Set it in .env file.")
    
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:1b")
    OLLAMA_URL = "http://localhost:11434"
    
    
    def post_progress(document_id: str, status: str, progress: int = 0, **kwargs):
        """Post progress update to progress tracking service."""
        try:
            payload = {"documentId": document_id, "status": status, "progress": progress, **kwargs}
            threading.Thread(target=lambda: requests.post(f"{PROGRESS_SERVICE_URL}/progress", json=payload, timeout=10), daemon=True).start()
        except:
            pass
    
    PERSIST_BASE = os.path.abspath("./chroma_store")
    SUMMARY_STORE_PATH = os.path.join(PERSIST_BASE, "summary_store")
    DETAILED_STORE_PATH = os.path.join(PERSIST_BASE, "detailed_store")
    IMAGE_STORE = os.path.abspath("./image_store")
    
    os.makedirs(SUMMARY_STORE_PATH, exist_ok=True)
    os.makedirs(DETAILED_STORE_PATH, exist_ok=True)
    os.makedirs(IMAGE_STORE, exist_ok=True)
    
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    
    EMBEDDINGS_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    GLOBAL_RERANKER = FlashrankRerank(top_n=5) # Default re-ranker
    
    def safe_metadata_value(value):
        """Convert unsupported metadata types (lists, dicts, objects) to JSON strings or plain strings.
    
        Chroma/Chromadb requires metadata values to be primitive types (str, int, float, bool, None) or SparseVector.
        We ensure we never store lists/dicts directly by serializing them.
        """
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        try:
            if isinstance(value, (list, dict)):
                return json.dumps(value)
        except Exception:
            pass
        try:
            return str(value)
        except Exception:
            return None
    
    
    def sanitize_metadata(metadata: dict) -> dict:
        """Return a sanitized metadata dict where every value is a primitive or None.
    
        - If metadata is not a dict, returns empty dict.
        - For nested dict/list, serializes to JSON string.
        """
        if not isinstance(metadata, dict):
            return {}
        sanitized = {}
        for k, v in metadata.items():
            try:
                sanitized[k] = safe_metadata_value(v)
            except Exception:
                sanitized[k] = None
        return sanitized
    
    def is_image_file(filename: str) -> bool:
        try:
            return Path(filename).suffix.lower() in IMAGE_EXTENSIONS
        except Exception:
            return False
    
    def get_public_url() -> str:
        if os.environ.get("PUBLIC_URL"):
            return os.environ.get("PUBLIC_URL")
        return globals().get("public_url", "http://localhost:8000")
    
    
    class Reference(BaseModel):
        """Pydantic model for a single citation reference."""
        id: str = Field(..., description="The citation ID, e.g., '1', '2'.")
        title: str = Field(..., description="The title of the source document.")
        source: str = Field(..., description="The source URL or filename.")
        page: int = Field(default=0, description="The page number in the document.")
        snippet: str = Field(default="", description="A short text snippet from the source.")
    
    class AIAnswer(BaseModel):
        """Pydantic model for the LLM's structured answer."""
        answer: str = Field(..., description="The detailed answer to the user's query, with IEEE-style citations like [1], [2].")
        references: List[Reference] = Field(..., description="A list of Reference objects used in the answer.")
    
    
    def deduplicate_references_and_update_answer(answer: str, references: List[Reference]) -> tuple[str, List[Reference]]:
        """
        Deduplicates references and updates the answer text to maintain correct reference numbering.
        """
        if not references:
            return answer, []
    
        unique_refs = {}
        id_mapping = {}
    
        new_id_counter = 1
        for ref in references:
            if ref.source not in unique_refs:
                new_id = str(new_id_counter)
                unique_refs[ref.source] = Reference(id=new_id, title=ref.title, source=ref.source, page=ref.page, snippet=ref.snippet)
                new_id_counter += 1
    
            id_mapping[ref.id] = unique_refs[ref.source].id
    
        updated_answer = answer
    
        sorted_old_ids = sorted(id_mapping.keys(), key=len, reverse=True)
    
        for old_id in sorted_old_ids:
            new_id = id_mapping[old_id]
            updated_answer = updated_answer.replace(f'[{old_id}]', f'[{new_id}]')
    
        return updated_answer, list(unique_refs.values())
    
    
    def stream_logs(proc, name):
        for line in iter(proc.stdout.readline, b''):
            sys.stdout.write(f"[{name}] {line.decode()}")
            sys.stdout.flush()
        for line in iter(proc.stderr.readline, b''):
            sys.stdout.write(f"[{name}-ERR] {line.decode()}")
            sys.stdout.flush()
    
    def start_ollama_service():
        ollama_proc = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=stream_logs, args=(ollama_proc, "Ollama"), daemon=True).start()
        print("Starting Ollama service...")
        for _ in range(40):
            try:
                if requests.get(OLLAMA_URL).status_code == 200:
                    print("Ollama is running locally!\n")
                    return True
            except:
                time.sleep(2)
        raise RuntimeError("Ollama failed to start in time.")
    
    def generate_image_summary(image_path: str, model: str = "llava") -> str:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": "Describe this image in detail, including any text, objects, colors, and context.", "images": [image_data], "stream": False},
                timeout=120
            )
            if response.status_code == 200:
                return response.json().get("response", "No description available")
            else:
                return f"Error generating summary: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"    # ==============================================================================
    
    def get_llm(model_name: str):
        """Unified function to get an LLM instance."""
        return OllamaLLM(model=model_name, format="json", temperature=0)
    
    def generate_with_llm(prompt: str, model_name: str):
        """Unified function to invoke an LLM for *non-structured* text."""
        llm = OllamaLLM(model=model_name, temperature=0.1)
    
        resp = llm.invoke(prompt)
        return getattr(resp, "content", str(resp))
    
    def _format_chunks_for_prompt(chunks: List[Document]) -> str:
        """Formats retrieved chunks into the string format"""
        context_strings = []
        for i, chunk in enumerate(chunks):
            if not hasattr(chunk, 'metadata') or not isinstance(chunk.metadata, dict):
                print(f"Warning: chunk {i} has invalid metadata, skipping")
                continue
            source = chunk.metadata.get("source", "N/A")
            title = chunk.metadata.get("title", chunk.metadata.get("filename", source))
            page = chunk.metadata.get("page", chunk.metadata.get("page_number", "N/A"))
    
            chunk_title = f"{title} (Page {page}, Chunk {i+1})"
            chunk.metadata["sense_title"] = chunk_title
    
            content = (
                f"=======================================DOCUMENT METADATA====================================\n"
                f"Source: {source}\n"
                f"Title: {chunk_title}\n"
                f"============================DOCUMENT PAGE CONTENT CHUNK=====================================\n"
                f"Page Content Chunk: \n\n{chunk.page_content}\n\n"
                f"====================================================================================="
            )
            context_strings.append(content)
        return "\n\n".join(context_strings)
    
    def build_advanced_rag_prompt(question: str, context: str) -> str:
        """Builds an advanced few-shot RAG prompt with IEEE-style citations."""
    
        return f"""
    You are a highly advanced AI research assistant. Your task is to answer the user's query based *only* on the provided document chunks.
    
    **Instructions:**
    1.  Read the **USER QUERY** carefully.
    2.  Analyze the **PROVIDED DOCUMENTS** to find all relevant information.
    3.  Synthesize a comprehensive, detailed answer that directly addresses the query.
    4.  **Crucially, you must cite your answer** using IEEE-style in-text citations, like [1], [2].
    5.  The **Title** and **Source** from the `DOCUMENT METADATA` must be used for citations.
    6.  You will be forced to output your response as a JSON object with an "answer" and "references" field.
    7.  Generate a reference for *every* piece of information you use.
    8.  **DO NOT** make up information. If the documents do not contain the answer, state that.
    
    ---
    **EXAMPLE OF HOW TO CITE:**
    
    **[Example] Provided Documents:**
    =======================================DOCUMENT METADATA====================================
    Source: https://example.com/ai.pdf
    Title: AI in 2024 (Page 5, Chunk 1)
    ============================DOCUMENT PAGE CONTENT CHUNK=====================================
    Page Content Chunk: \n\nArtificial intelligence has seen
    rapid growth, especially in large language models. [1]
    =====================================================================================
    =======================================DOCUMENT METADATA====================================
    Source: https://example.com/ml.pdf
    Title: ML Basics (Page 2, Chunk 4)
    ============================DOCUMENT PAGE CONTENT CHUNK=====================================
    Page Content Chunk: \n\nMachine learning is a subset of AI.
    =====================================================================================
    
    **[Example] Expected JSON Output:**
    {{
      "answer": "Artificial intelligence (AI) has experienced rapid growth, particularly in the realm of large language models [1]. Machine learning is known to be a subset of AI [2].",
      "references": [
    {{
      "id": "1",
      "title": "AI in 2024 (Page 5, Chunk 1)",
      "source": "https://example.com/ai.pdf",
      "page": 5,
      "snippet": "Artificial intelligence has seen rapid growth, especially in large language models."
    }},
    {{
      "id": "2",
      "title": "ML Basics (Page 2, Chunk 4)",
      "source": "https://example.com/ml.pdf",
      "page": 2,
      "snippet": "Machine learning is a subset of AI."
    }}
      ]
    }}
    ---
    
    **ACTUAL TASK:**
    
    **USER QUERY:** {question}
    
    **PROVIDED DOCUMENTS:**
    {context}
    
    **YOUR JSON RESPONSE:**
    """
    
    
    class HierarchicalRAGService:
        """
        Manages the Hierarchical Vector Stores (Summary & Detailed)
        and all core RAG logic.
        """
        def __init__(self, summary_path, detailed_path, embeddings):
            self.embeddings = embeddings
            self.summary_store = Chroma(
                collection_name="summary_store",
                embedding_function=self.embeddings,
                persist_directory=summary_path
            )
            self.detailed_store = Chroma(
                collection_name="detailed_store",
                embedding_function=self.embeddings,
                persist_directory=detailed_path
            )
    
        def _load_and_split_pdf(self, pdf_bytes: bytes) -> List[Document]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                path = tmp.name
            docs = PyMuPDFLoader(path).load()
            if not docs:
                os.remove(path)
                return []
    
            pdf = fitz.open(path)
            images_per_page = {}
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                images = page.get_images(full=True)
                page_images = []
                for img_index, img in enumerate(images):
                    xref = img[0]
                    try:
                        base_image = pdf.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        img_id = f"img_{uuid4().hex}"
                        image_filename = f"{img_id}.{image_ext}"
                        image_path = os.path.join(IMAGE_STORE, image_filename)
                        with open(image_path, "wb") as f:
                            f.write(image_bytes)
                        summary = generate_image_summary(image_path)
                        image_url = f"{get_public_url().rstrip('/')}/image_bytes/{img_id}"
                        page_images.append({
                            "id": img_id,
                            "url": image_url,
                            "summary": summary,
                            "page": page_num + 1,
                            "ext": image_ext
                        })
                    except Exception as e:
                        pass
                images_per_page[page_num + 1] = page_images
            pdf.close()
            os.remove(path)
    
            new_docs = []
            for doc in docs:
                page_num = getattr(doc, 'metadata', {}).get('page', 1) if hasattr(doc, 'metadata') else 1
                images = images_per_page.get(page_num, [])
                if not isinstance(doc, Document):
                    doc = Document(page_content=str(doc), metadata={})
                if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
                    doc.metadata = {}
                doc.metadata["images"] = images
                new_docs.append(doc)
    
            splitter = SemanticChunker(self.embeddings, breakpoint_threshold_type="percentile")
            chunks = splitter.split_documents(new_docs)
            safe_chunks = []
            for c in chunks:
                if isinstance(c, Document):
                    safe_chunks.append(c)
                else:
                    safe_chunks.append(Document(page_content=str(c), metadata={}))
            return safe_chunks
    
        async def _generate_summary_for_ingestion(self, chunks: List[Document], model_name: str) -> str:
            if not chunks: return "No text content found."
            semaphore = asyncio.Semaphore(10)
            async def summarize_chunk_async(chunk_text: str) -> str:
                prompt = f"Summarize the following text chunk in 2-3 key bullet points:\n\n{chunk_text}\n\nSummary:"
                async with semaphore:
                    return await asyncio.to_thread(generate_with_llm, prompt, model_name)
            tasks = [asyncio.create_task(summarize_chunk_async(c.page_content)) for c in chunks]
            intermediate_summaries = await asyncio.gather(*tasks)
            combined_summaries = "\n".join(intermediate_summaries)
            synthesis_prompt = (
                f"Create a single, concise paragraph summarizing the key themes "
                f"from the following list of chunk summaries.\n\nSummaries:\n{combined_summaries}\n\nOverall Summary Paragraph:"
            )
            final_summary = await asyncio.to_thread(generate_with_llm, synthesis_prompt, model_name)
            return final_summary
    
        async def add_document_to_stores(self, pdf_bytes: bytes, doc_id: str, model_name: str):
            post_progress(doc_id, "loading", 5, message="Loading PDF...")
    
            chunks = self._load_and_split_pdf(pdf_bytes)
            try:
                print(f"DEBUG: Received {len(chunks)} chunks from splitter")
                for i, c in enumerate(chunks[:5]):
                    print(f"DEBUG chunk {i}: type={type(c)}, has_metadata={hasattr(c, 'metadata')}, metadata_type={type(getattr(c, 'metadata', None))}")
            except Exception as e:
                print(f"DEBUG: Could not inspect chunks: {e}")
            if not chunks:
                post_progress(doc_id, "failed", 0, message="No text content found")
                raise ValueError("No text content found in the document")
    
            post_progress(doc_id, "chunking", 20, message=f"Split into {len(chunks)} chunks", totalChunks=len(chunks))
    
            if chunks and hasattr(chunks[0], 'metadata') and isinstance(chunks[0].metadata, dict):
                source_filename = chunks[0].metadata.get("source", f"doc_{doc_id}")
            else:
                print(f"Warning: first chunk missing metadata or invalid type ({type(chunks[0])}), using fallback source filename")
                source_filename = f"doc_{doc_id}"
    
            post_progress(doc_id, "summarizing", 40, message="Generating document summary...", totalChunks=len(chunks))
            summary_text = await self._generate_summary_for_ingestion(chunks, model_name)
    
            post_progress(doc_id, "embedding_summary", 60, message="Creating summary embeddings...", totalChunks=len(chunks))
            summary_doc = Document(
                page_content=summary_text,
                metadata={"doc_id": doc_id, "source": source_filename, "title": f"Summary for {source_filename}"}
            )
            self.summary_store.add_documents([summary_doc], ids=[doc_id])
    
            post_progress(doc_id, "embedding_chunks", 75, message="Creating chunk embeddings...", totalChunks=len(chunks))
            for i, chunk in enumerate(chunks):
                if not isinstance(chunk, Document):
                    print(f"Converting non-Document chunk at index {i} of type {type(chunk)} to Document")
                    chunk = Document(page_content=str(chunk), metadata={})
                    chunks[i] = chunk
                if not hasattr(chunk, 'metadata') or not isinstance(chunk.metadata, dict):
                    chunk.metadata = {}
                chunk.metadata["doc_id"] = doc_id
                chunk.metadata["title"] = f"{Path(source_filename).name} (Page {chunk.metadata.get('page', i+1)})"
                try:
                    chunk.metadata = sanitize_metadata(chunk.metadata)
                except Exception as e:
                    print(f"Warning: sanitize_metadata failed for chunk {i}: {e}; using unfiltered metadata")
                if i % 5 == 0 or i == len(chunks) - 1:
                    progress = 75 + int((i + 1) / len(chunks) * 20)
                    post_progress(doc_id, "embedding_chunks", progress,
                                message=f"Embedding chunk {i+1}/{len(chunks)}...",
                                currentChunk=i+1, totalChunks=len(chunks))
    
            try:
                for i,c in enumerate(chunks[:5]):
                    print(f"DEBUG before add_documents chunk {i} metadata types: {[type(v) for v in c.metadata.values()]} keys={list(c.metadata.keys())}")
            except Exception as e:
                print(f"DEBUG: failed to print chunk metadata diagnostics: {e}")
            for i, ch in enumerate(chunks):
                if not hasattr(ch, 'metadata') or not isinstance(ch.metadata, dict):
                    ch.metadata = {}
                try:
                    ch.metadata = sanitize_metadata(ch.metadata)
                except Exception as e:
                    print(f"Warning: final sanitize_metadata failed for chunk {i}: {e}; using empty metadata")
                    ch.metadata = {}
            chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            try:
                self.detailed_store.add_documents(chunks, ids=chunk_ids)
            except Exception as e:
                print("ERROR: adding documents to detailed_store failed", e)
                try:
                    for i, ch in enumerate(chunks[:10]):
                        print(f"DEBUG add failure chunk {i} metadata types: {[type(v) for v in ch.metadata.values()]} metadata_preview={json.dumps(ch.metadata)[:200]}")
                except Exception:
                    pass
                post_progress(doc_id, "failed", 0, message=f"Failed to add documents: {e}")
                raise
    
            post_progress(doc_id, "complete", 100, message="Document processing complete!", totalChunks=len(chunks))
            return len(chunks)
    
        def get_chunks_by_doc_id(self, doc_id: str) -> List[Document]:
            results = self.detailed_store.get(where={"doc_id": doc_id}, include=["metadatas", "documents"])
            print(f"DEBUG: detailed_store.get returned keys: {list(results.keys())}")
            try:
                print(f"DEBUG: metadatas type={type(results.get('metadatas'))}, documents type={type(results.get('documents'))}")
                if isinstance(results.get('documents'), list):
                    for i, d in enumerate(results.get('documents')[:5]):
                        print(f"DEBUG doc {i} type={type(d)} content_preview={str(d)[:80]}")
            except Exception as e:
                print(f"DEBUG: could not inspect results from detailed_store.get: {e}")
            if not results.get('documents') or not results.get('metadatas'):
                return []
            docs = []
            for i, text in enumerate(results['documents']):
                if i >= len(results['metadatas']):
                    continue
                meta = results['metadatas'][i]
                try:
                    if isinstance(text, str) and isinstance(meta, dict):
                        sanitized_meta = sanitize_metadata(meta)
                        docs.append(Document(page_content=text, metadata=sanitized_meta))
                    else:
                        print(f"Warning: unexpected type for document {i}: text={type(text)}, meta={type(meta)}")
                except Exception as e:
                    print(f"Error creating document {i}: {e}")
                    continue
            return docs
    
        async def query_rag(self, document_ids: List[str], question: str, model_name: str, top_k: int = 5, specific_chunks: Dict[str, List[int]] = None) -> Tuple[str, List[Dict[str, Any]]]:
    
            summary_retriever = self.summary_store.as_retriever(search_kwargs={'k': 20, 'filter': {'doc_id': {'$in': document_ids}}})
            summary_compressor = ContextualCompressionRetriever(base_compressor=GLOBAL_RERANKER, base_retriever=summary_retriever)
            relevant_summaries = summary_compressor.invoke(question)
            relevant_doc_ids = []
            for doc in relevant_summaries:
                if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict) and 'doc_id' in doc.metadata:
                    relevant_doc_ids.append(doc.metadata['doc_id'])
            relevant_doc_ids = list(set(relevant_doc_ids))
    
            if not relevant_doc_ids:
                return "No relevant documents found.", []
    
            if specific_chunks:
                relevant_chunks = []
                for doc_id in relevant_doc_ids:
                    if doc_id in specific_chunks:
                        all_chunks = self.get_chunks_by_doc_id(doc_id)
                        selected_indices = specific_chunks[doc_id]
                        for idx in selected_indices:
                            if idx < len(all_chunks):
                                relevant_chunks.append(all_chunks[idx])
            else:
                detailed_retriever = self.detailed_store.as_retriever(search_kwargs={'k': 25, 'filter': {'doc_id': {'$in': relevant_doc_ids}}})
                chunk_compressor = ContextualCompressionRetriever(base_compressor=FlashrankRerank(top_n=top_k), base_retriever=detailed_retriever)
                relevant_chunks = chunk_compressor.invoke(question)
                for i, rc in enumerate(relevant_chunks):
                    if not isinstance(rc, Document):
                        print(f"Converting non-Document relevant chunk at index {i} of type {type(rc)}")
                        relevant_chunks[i] = Document(page_content=str(rc), metadata={})
                print(f"DEBUG: relevant_chunks types after conversion: {[type(rc) for rc in relevant_chunks[:5]]}")
    
            if not relevant_chunks:
                return "No relevant chunks found.", []
    
            context_string = _format_chunks_for_prompt(relevant_chunks)
            final_prompt = build_advanced_rag_prompt(question, context_string)
    
            llm_name_for_rag = model_name
            if model_name.lower() not in ["mistral", "llama3"]:
                llm_name_for_rag = "mistral"
    
            try:
                llm = get_llm(llm_name_for_rag)
    
                json_prompt = final_prompt + """\n\nReturn JSON: {"answer": "...", "references": [{"id": "1", "title": "...", "source": "...", "page": 1, "snippet": "..."}]}"""
                raw_response = await asyncio.to_thread(llm.invoke, json_prompt)
                raw_response_content = getattr(raw_response, "content", str(raw_response))
                print(f"✓ LLM response received (length: {len(raw_response_content)} chars)")
                print(f"📄 Raw response preview: {raw_response_content[:300]}...")
    
                import json, re
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_response_content, re.DOTALL) or re.search(r'\{.*"answer".*"references".*\}', raw_response_content, re.DOTALL)
                if not json_match:
                    raise ValueError("No JSON in response")
                parsed_json = json.loads(json_match.group(1) if json_match.groups() else json_match.group(0))
                references = [Reference(id=str(ref.get('id', i+1)), title=ref.get('title', 'Unknown'), source=ref.get('source', 'Unknown'), page=ref.get('page', 0), snippet=ref.get('snippet', '')) for i, ref in enumerate(parsed_json.get('references', []))]
                ai_answer_response = AIAnswer(answer=parsed_json.get('answer', ''), references=references)
    
                final_answer, final_refs = deduplicate_references_and_update_answer(ai_answer_response.answer, ai_answer_response.references)
    
                chunk_map = {}
                for chunk in relevant_chunks:
                    if hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict):
                        source = chunk.metadata.get("source", "")
                        if source:
                            chunk_map[source] = chunk
                final_refs_dict = []
                for i, ref in enumerate(final_refs):
                    chunk = chunk_map.get(ref.source)
                    page = ref.page or (chunk.metadata.get("page", 0) if chunk and hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict) else 0)
                    snippet = ref.snippet or (chunk.page_content[:200] if chunk else "")
                    full_text = chunk.page_content if chunk else ref.title
                    final_refs_dict.append({
                        "documentId": ref.source.split("/")[-1] if "/" in ref.source else ref.source,
                        "page": page,
                        "snippet": snippet,
                        "fullText": full_text,
                        "source": ref.title,
                        "rank": i + 1,
                        "score": None
                    })
    
                return final_answer, final_refs_dict
    
            except Exception as e:
                simple_context = "\n\n".join([c.page_content for c in relevant_chunks])
                simple_prompt = f"Answer: {question}\n\nContext:\n{simple_context}\n\nAnswer:"
                return generate_with_llm(simple_prompt, model_name), []
    
    
    
    print("Starting FastAPI app...")
    
    app = FastAPI(title="Hierarchical RAG API with Structured Citations")
    
    try:
        rag_service = HierarchicalRAGService(
            summary_path=SUMMARY_STORE_PATH,
            detailed_path=DETAILED_STORE_PATH,
            embeddings=EMBEDDINGS_MODEL
        )
    except Exception as e:
        print(f"FATAL: Could not initialize RAG Service: {e}")
        rag_service = None
    
    @app.on_event("startup")
    async def startup_event():
        if rag_service is None:
            pass
    
    @app.get("/health")
    def health_check():
        return {"status": "healthy", "message": "Backend is running"}
    
    @app.get("/")
    def home():
        if rag_service is None: raise HTTPException(status_code=500, detail="RAG Service is not operational.")
        return {
            "message": "Hierarchical RAG API with Structured Citations - Active",
            "summary_store_count": rag_service.summary_store._collection.count(),
            "detailed_store_count": rag_service.detailed_store._collection.count()
        }
    
    @app.post("/process")
    async def process(file: UploadFile):
        if rag_service is None: raise HTTPException(status_code=500, detail="RAG Service is not operational.")
        if is_image_file(file.filename):
            doc_id = f"img_{uuid4().hex}"
            image_path = os.path.join(IMAGE_STORE, f"{doc_id}{Path(file.filename).suffix}")
            with open(image_path, "wb") as f: f.write(await file.read())
            print(f"Image saved: {image_path}")
            return {"documentId": doc_id, "status": "image_saved", "isImage": True, "imagePath": image_path}
    
        doc_id = f"doc_{uuid4().hex}"
        try:
            pdf_bytes = await file.read()
            chunk_count = await rag_service.add_document_to_stores(pdf_bytes, doc_id, "mistral") # Use fast model for ingestion
            return {"documentId": doc_id, "status": "embeddings_created", "chunkCount": chunk_count, "isImage": False}
        except Exception as e:
            print(f"Error processing document: {e}")
            return JSONResponse(status_code=500, content={"error": "Failed to process document", "message": str(e)})
    
    @app.post("/get_chunks")
    async def get_chunks(request: Request):
        """Get all chunks for a specific document ID."""
        if rag_service is None: raise HTTPException(status_code=500, detail="RAG Service is not operational.")
        data = await request.json()
        document_id = data.get("documentId")
        if not document_id: raise HTTPException(status_code=400, detail="documentId is required")
        chunks = rag_service.get_chunks_by_doc_id(document_id)
        if not chunks: raise HTTPException(status_code=404, detail=f"No chunks found for documentId {document_id}")
        def _parse_images_field(md):
            try:
                if not md: return []
                imgs = md.get("images")
                if imgs is None:
                    return []
                if isinstance(imgs, str):
                    try:
                        parsed = json.loads(imgs)
                        return parsed if isinstance(parsed, list) else [parsed]
                    except Exception:
                        return []
                elif isinstance(imgs, list):
                    return imgs
                else:
                    return []
            except Exception:
                return []
    
        return JSONResponse(content={
            "documentId": document_id,
            "chunks": [
                {
                    "id": i,
                    "content": chunk.page_content,
                    "metadata": chunk.metadata if hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict) else {},
                    "images": _parse_images_field(chunk.metadata if hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict) else {})
                }
                for i, chunk in enumerate(chunks)
            ]
        })
    
    @app.post("/generate")
    async def generate_text(request: Request):
        """
        Main RAG endpoint with hierarchical retrieval and structured citations.
        """
        if rag_service is None: raise HTTPException(status_code=500, detail="RAG Service is not operational.")
    
        data = await request.json()
        model = data.get("model", OLLAMA_MODEL)
        prompt = data.get("prompt", "")
        document_ids = data.get("documentIds", [])
        specific_chunks = data.get("specificChunks", None)  # Optional: {"doc_id": [0, 2, 5], ...}
    
        image_ids = [doc_id for doc_id in document_ids if doc_id.startswith("img_")]
        text_ids = [doc_id for doc_id in document_ids if doc_id.startswith("doc_")]
    
        if image_ids:
            return await process_image_query(image_ids, text_ids, prompt, model)
    
        if text_ids:
            max_citations = 7 # Get more chunks for the advanced prompt
    
            response_text, citations = await rag_service.query_rag(
                document_ids=text_ids,
                question=prompt,
                model_name=model,
                specific_chunks=specific_chunks,
                top_k=max_citations
            )
            return JSONResponse(content={"response": response_text, "citations": citations})
    
        response_text = generate_with_llm(prompt, model) # Uses simple text gen
        return JSONResponse(content={"response": response_text, "citations": []})
    
    
    async def process_image_query(image_ids: list, text_ids: list, prompt: str, model: str):
        """
        Image Q&A function with optional RAG context from text documents.
        """
        vision_model = "llava"
        responses = []
    
        for img_id in image_ids:
            image_files = [f for f in os.listdir(IMAGE_STORE) if f.startswith(img_id)]
            if not image_files:
                responses.append(f"Image {img_id} not found.")
                continue
    
            image_path = os.path.join(IMAGE_STORE, image_files[0])
            with open(image_path, "rb") as f: image_data = base64.b64encode(f.read()).decode()
            try:
                response = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": vision_model, "prompt": prompt, "images": [image_data], "stream": False},
                    timeout=120
                )
                if response.status_code == 200: responses.append(response.json().get("response", "No response"))
                else: responses.append(f"Error: Vision model status {response.status_code}")
            except Exception as e: responses.append(f"Error processing image: {str(e)}")
    
        additional_context = ""
        citations = []
    
        if text_ids and rag_service:
            print("... Image query also performing RAG on text documents ...")
            context, citations = await rag_service.query_rag(text_ids, prompt, model, top_k=3)
            additional_context = f"\n\nAdditional context from documents:\n{context}"
    
        final_response = "\n\n".join(responses)
        if additional_context: final_response += additional_context
    
        return JSONResponse(content={"response": final_response, "citations": citations, "usedVisionModel": True, "visionModel": vision_model})
    
    
    @app.get("/image_bytes/{image_id}")
    async def get_image_bytes(image_id: str):
        """Serve an image by its ID as bytes."""
        for ext in IMAGE_EXTENSIONS:
            image_path = os.path.join(IMAGE_STORE, f"{image_id}{ext}")
            if os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    data = f.read()
                return Response(data, media_type=f"image/{ext[1:]}")
        raise HTTPException(status_code=404, detail="Image not found")
    
    
    @app.post("/pull")
    async def pull_model(request: Request):
        try:
            model = (await request.json()).get("name", OLLAMA_MODEL)
            resp = requests.post(f"{OLLAMA_URL}/api/pull", json={"name": model, "stream": False}, timeout=300)
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})
    
    
    try:
        start_ollama_service()
        os.system("ollama pull gemma3:1b && ollama pull llava")
    except:
        pass
    
    FIXED_URL = "https://mari-unbequeathed-milkily.ngrok-free.app"
    public_url = "http://localhost:8000"
    ngrok_enabled = False
    
    if NGROK_AUTHTOKEN and NGROK_AUTHTOKEN != "YOUR_NGROK_AUTHTOKEN":
        os.system(f"ngrok config add-authtoken {NGROK_AUTHTOKEN}")
        ngrok_proc = subprocess.Popen(["ngrok", "http", "--host-header=rewrite", "--url", FIXED_URL, "8000"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        public_url = FIXED_URL
        ngrok_enabled = True
        time.sleep(3)
        print(f"Public URL: {public_url}")
    
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)
    server = uvicorn.Server(config)
    
    threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True).start()
    
    try:
        while True: time.sleep(300)
    except KeyboardInterrupt:
        if ngrok_enabled and ngrok_proc:
            ngrok_proc.terminate()
    
if __name__ == "__main__":
    main()
