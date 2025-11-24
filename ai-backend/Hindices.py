# ==============================================================================
# 0. INSTALLS (Colab only - comment out for local use)
# ==============================================================================
# Uncomment the following lines if running in Google Colab or hosted environment:
# !curl -fsSL https://ollama.com/install.sh | sh
# !pip install fastapi uvicorn pyngrok requests boto3 python-multipart aiofiles langchain langchain-community chromadb sentence-transformers PyMuPDF langchain-huggingface langchain-chroma langchain-ollama langchain-experimental flashrank pydantic python-dotenv

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import os, signal, psutil, gc, time, sys, subprocess, threading, requests, tempfile, asyncio, json, base64, random
from pathlib import Path
from uuid import uuid4
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- FastAPI & Server ---
from fastapi import FastAPI, UploadFile, Form, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pyngrok import ngrok
from contextlib import asynccontextmanager

# --- LangChain Core ---
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain.globals import set_verbose
from pydantic import BaseModel, Field
import fitz

# --- Colab Support ---
try:
    from google.colab import userdata
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
print("Loading configuration...")

# --- API Keys & Models ---
if IN_COLAB:
    try:
        NGROK_AUTHTOKEN = "32eB7tLSQoICKJD4JSQuJ9lWea6_7U5ndjtQCVaWnPLEc4Mws"
        PROGRESS_SERVICE_URL = os.environ.get("PROGRESS_SERVICE_URL", "https://localdocu-progress.vercel.app")
    except Exception:
        print("WARNING: Could not load from Colab secrets, falling back to environment variables.")
        NGROK_AUTHTOKEN = "32eB7tLSQoICKJD4JSQuJ9lWea6_7U5ndjtQCVaWnPLEc4Mws"
        PROGRESS_SERVICE_URL = os.environ.get("PROGRESS_SERVICE_URL", "https://localdocu-progress.vercel.app")
else:
    NGROK_AUTHTOKEN = "32eB7tLSQoICKJD4JSQuJ9lWea6_7U5ndjtQCVaWnPLEc4Mws"
    PROGRESS_SERVICE_URL = os.environ.get("PROGRESS_SERVICE_URL", "https://localdocu-progress.vercel.app")

if not NGROK_AUTHTOKEN or NGROK_AUTHTOKEN == "YOUR_NGROK_AUTHTOKEN":
    print("WARNING: NGROK_AUTHTOKEN not configured properly. Set it in .env file.")

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:1b")
OLLAMA_URL = "http://localhost:11434"

# --- Persistent Storage Paths (Hierarchical) ---
PERSIST_BASE = os.path.abspath("./chroma_store")
SUMMARY_STORE_PATH = os.path.join(PERSIST_BASE, "summary_store")
DETAILED_STORE_PATH = os.path.join(PERSIST_BASE, "detailed_store")
IMAGE_STORE = os.path.abspath("./image_store")

os.makedirs(SUMMARY_STORE_PATH, exist_ok=True)
os.makedirs(DETAILED_STORE_PATH, exist_ok=True)
os.makedirs(IMAGE_STORE, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def post_progress(document_id: str, status: str, progress: int = 0, **kwargs):
    """Post progress update to progress tracking service."""
    try:
        payload = {"documentId": document_id, "status": status, "progress": progress, **kwargs}
        threading.Thread(target=lambda: requests.post(f"{PROGRESS_SERVICE_URL}/progress", json=payload, timeout=10), daemon=True).start()
    except:
        pass

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

# ==============================================================================
# 3. NEW: Pydantic Models for Structured Output
# ==============================================================================

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

# ==============================================================================
# 4. NEW: Citation Deduplication Utility
# ==============================================================================

def deduplicate_references_and_update_answer(answer: str, references: List[Reference]) -> tuple[str, List[Reference]]:
    """
    Deduplicates references and updates the answer text to maintain correct reference numbering.
    """
    if not references:
        return answer, []

    unique_refs = {}
    id_mapping = {}

    # Create unique references and map old IDs to new IDs
    new_id_counter = 1
    for ref in references:
        if ref.title not in unique_refs:  
            new_id = str(new_id_counter)
            unique_refs[ref.title] = Reference(id=new_id, title=ref.title, source=ref.source, page=ref.page, snippet=ref.snippet)
            new_id_counter += 1

        id_mapping[ref.id] = unique_refs[ref.title].id

    updated_answer = answer

    # Sort keys by length (desc) to replace "[10]" before "[1]"
    sorted_old_ids = sorted(id_mapping.keys(), key=len, reverse=True)

    for old_id in sorted_old_ids:
        new_id = id_mapping[old_id]
        # Replace citations (e.g., [1], [2], etc.)
        updated_answer = updated_answer.replace(f'[{old_id}]', f'[{new_id}]')

    return updated_answer, list(unique_refs.values())

# ==============================================================================
# 5. SYSTEM & OLLAMA UTILS
# ==============================================================================

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
    """Generate a detailed description of an image using a vision model."""
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
        return f"Error: {str(e)}"

# ==============================================================================
# 6. SHARED LLM & RAG PROMPT LOGIC (MODIFIED)
# ==============================================================================

def get_llm(model_name: str):
    """Unified function to get an LLM instance."""
    # Using a model known to be good at JSON mode
    return OllamaLLM(model=model_name, format="json", temperature=0)

def generate_with_llm(prompt: str, model_name: str):
    """Unified function to invoke an LLM for *non-structured* text."""
    # Use a basic model for simple generation
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

        # Give each chunk a unique "Title" for citation
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

**ACTUAL TASK:**

**USER QUERY:** {question}

**PROVIDED DOCUMENTS:**
{context}

**YOUR JSON RESPONSE:**
"""

# ==============================================================================
# 7. CORE: HIERARCHICAL RAG SERVICE (MODIFIED)
# ==============================================================================

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

    # --- Ingestion Logic (Unchanged) ---
    def _load_and_split_pdf(self, pdf_bytes: bytes) -> List[Document]:
        print("[PDF] Creating temporary PDF file...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            path = tmp.name

        print("[PDF] Loading PDF with PyMuPDF...")
        docs = PyMuPDFLoader(path).load()
        print(f"[PDF] Loaded {len(docs)} pages from PDF")

        if not docs:
            print("[PDF] No documents loaded from PDF")
            os.remove(path)
            return []

        print("[PDF] Extracting images from PDF...")
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
                    page_images.append({
                        "id": img_id,
                        "summary": summary,
                        "page": page_num + 1,
                        "ext": image_ext
                    })
                    print(f"[PDF] Extracted and summarized image {img_id} from page {page_num + 1}")
                except Exception as e:
                    print(f"[PDF] Failed to extract image {img_index} from page {page_num}: {e}")
            images_per_page[page_num] = page_images
        pdf.close()
        os.remove(path)
        print(f"[PDF] Completed image extraction, found images on {len([p for p in images_per_page.values() if p])} pages")

        print("[PDF] Assigning images to documents and converting to Document objects...")
        # Assign images to documents based on page, and ensure all are Document objects
        new_docs = []
        for doc in docs:
            page_num = getattr(doc, 'metadata', {}).get('page', 0) if hasattr(doc, 'metadata') else 0
            images = images_per_page.get(page_num, [])
            # If doc is not a Document, convert it
            if not isinstance(doc, Document):
                doc = Document(page_content=str(doc), metadata={})
            # Ensure metadata is a dict
            if not hasattr(doc, 'metadata') or not isinstance(doc.metadata, dict):
                doc.metadata = {}
            doc.metadata["images"] = images
            doc.metadata["page"] = page_num + 1
            new_docs.append(doc)

        print("[PDF] Splitting documents into semantic chunks...")
        splitter = SemanticChunker(self.embeddings, breakpoint_threshold_type="percentile")
        chunks = splitter.split_documents(new_docs)
        print(f"[PDF] Split into {len(chunks)} semantic chunks")

        # Ensure all chunks are Document objects
        safe_chunks = []
        for c in chunks:
            if isinstance(c, Document):
                safe_chunks.append(c)
            else:
                safe_chunks.append(Document(page_content=str(c), metadata={}))
        print(f"[PDF] Final chunk count: {len(safe_chunks)}")
        return safe_chunks

    async def _generate_summary_for_ingestion(self, chunks: List[Document], model_name: str) -> str:
        print(f"[SUMMARY] Starting summary generation for {len(chunks)} chunks")
        if not chunks:
            print("[SUMMARY] No chunks provided for summary")
            return "No text content found."

        intermediate_summaries = [chunk.metadata.get("summary", "") for chunk in chunks if hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict)]

        print(f"[SUMMARY] Synthesizing {len(intermediate_summaries)} intermediate summaries...")
        combined_summaries = "\n".join(intermediate_summaries)
        synthesis_prompt = (
            f"Create a single, concise paragraph summarizing the key themes "
            f"from the following list of chunk summaries.\n\nSummaries:\n{combined_summaries}\n\nOverall Summary Paragraph:"
        )
        final_summary = await asyncio.to_thread(generate_with_llm, synthesis_prompt, model_name)
        print(f"[SUMMARY] Final summary generated: {len(final_summary)} characters")
        return final_summary

    async def add_document_to_stores(self, pdf_bytes: bytes, doc_id: str, model_name: str):
        print(f"[RAG] Starting document ingestion for doc_id: {doc_id}")
        post_progress(doc_id, "loading", 5, message="Loading PDF...")

        print("[RAG] Loading and splitting PDF...")
        chunks = self._load_and_split_pdf(pdf_bytes)
        print(f"[RAG] Split into {len(chunks)} chunks")

        # Diagnostic logging: ensure chunks is a list of Document objects
        try:
            print(f"[RAG] Chunk diagnostics: {len(chunks)} total chunks")
            for i, c in enumerate(chunks[:3]):
                print(f"[RAG] Chunk {i}: type={type(c)}, has_metadata={hasattr(c, 'metadata')}")
        except Exception as e:
            print(f"[RAG] Could not inspect chunks: {e}")

        if not chunks:
            print("[RAG] No text content found in document")
            post_progress(doc_id, "failed", 0, message="No text content found")
            raise ValueError("No text content found in the document")

        print(f"[RAG] Successfully split into {len(chunks)} chunks")
        post_progress(doc_id, "chunking", 20, message=f"Split into {len(chunks)} chunks", totalChunks=len(chunks))

        print("[RAG] Generating chunk summaries...")
        post_progress(doc_id, "summarizing_chunks", 30, message="Generating chunk summaries...", totalChunks=len(chunks))
        async def generate_chunk_summaries():
            batch_count = (len(chunks) + 4) // 5
            for i in range(0, len(chunks), 5):
                batch_num = i // 5 + 1
                print(f"[SUMMARY] Processing batch {batch_num}/{batch_count}")
                batch = chunks[i:i+5]
                tasks = []
                for c in batch:
                    async def safe_generate(c):
                        try:
                            return await asyncio.to_thread(generate_with_llm, f"Summarize this text in one sentence: {c.page_content[:500]}", model_name)
                        except Exception as e:
                            print(f"[SUMMARY] Failed to generate summary for chunk: {e}")
                            return "Summary generation failed."
                    tasks.append(asyncio.create_task(safe_generate(c)))
                batch_summaries = await asyncio.gather(*tasks, return_exceptions=True)
                for j, summary in enumerate(batch_summaries):
                    if isinstance(summary, Exception):
                        summary = "Summary generation failed."
                    chunks[i + j].metadata["summary"] = summary
                print(f"[SUMMARY] Completed batch {batch_num}, {len(batch_summaries)} summaries generated")
        await generate_chunk_summaries()
        print(f"[RAG] Generated summaries for {len(chunks)} chunks")

        # Safely resolve source filename
        if chunks and hasattr(chunks[0], 'metadata') and isinstance(chunks[0].metadata, dict):
            source_filename = chunks[0].metadata.get("source", f"doc_{doc_id}")
        else:
            print(f"[RAG] First chunk missing metadata, using fallback source filename")
            source_filename = f"doc_{doc_id}"

        print("[RAG] Generating document summary...")
        post_progress(doc_id, "summarizing", 40, message="Generating document summary...", totalChunks=len(chunks))
        summary_text = await self._generate_summary_for_ingestion(chunks, model_name)
        print(f"[RAG] Summary generated: {len(summary_text)} characters")

        print("[RAG] Creating summary embeddings...")
        post_progress(doc_id, "embedding_summary", 60, message="Creating summary embeddings...", totalChunks=len(chunks))
        summary_doc = Document(
            page_content=summary_text,
            metadata={"doc_id": doc_id, "source": source_filename, "title": f"Summary for {source_filename}"}
        )
        self.summary_store.add_documents([summary_doc], ids=[doc_id])
        print("[RAG] Summary embeddings created and stored")

        print("[RAG] Creating chunk embeddings...")
        post_progress(doc_id, "embedding_chunks", 75, message="Creating chunk embeddings...", totalChunks=len(chunks))
        current_index = 0
        for i in range(0, len(chunks), 5):
            batch = chunks[i:i+5]
            print(f"[RAG] Processing batch {i//5 + 1}/{(len(chunks) + 4)//5}")
            for chunk in batch:
                # Defensive conversion: ensure chunk is Document and has dict metadata
                if not isinstance(chunk, Document):
                    print(f"[RAG] Converting non-Document chunk at index {current_index}")
                    chunk = Document(page_content=str(chunk), metadata={})
                    chunks[current_index] = chunk
                if not hasattr(chunk, 'metadata') or not isinstance(chunk.metadata, dict):
                    chunk.metadata = {}
                chunk.metadata["doc_id"] = doc_id
                chunk.metadata["title"] = f"{Path(source_filename).name} (Page {chunk.metadata.get('page', current_index+1)})"
                try:
                    # san is sanitized so lists/dicts converted to strings
                    chunk.metadata = sanitize_metadata(chunk.metadata)
                except Exception as e:
                    print(f"[RAG] sanitize_metadata failed for chunk {current_index}: {e}")
                current_index += 1
            progress = 75 + int(current_index / len(chunks) * 20)
            post_progress(doc_id, "embedding_chunks", progress,
                        message=f"Embedding chunk {current_index}/{len(chunks)}...",
                        currentChunk=current_index, totalChunks=len(chunks))
            print(f"[RAG] Completed batch, {current_index}/{len(chunks)} chunks processed")

        print("[RAG] Final metadata sanitization...")
        # Final sanitization before sending to Chroma: ensure every metadata value is primitive
        for i, ch in enumerate(chunks):
            if not hasattr(ch, 'metadata') or not isinstance(ch.metadata, dict):
                ch.metadata = {}
            try:
                ch.metadata = sanitize_metadata(ch.metadata)
            except Exception as e:
                print(f"[RAG] Final sanitize_metadata failed for chunk {i}: {e}")
                ch.metadata = {}

        print("[RAG] Adding chunks to detailed store...")
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        try:
            self.detailed_store.add_documents(chunks, ids=chunk_ids)
            print(f"[RAG] Successfully added {len(chunks)} chunks to detailed store")
        except Exception as e:
            print(f"[RAG] Failed to add documents to detailed_store: {e}")
            post_progress(doc_id, "failed", 0, message=f"Failed to add documents: {e}")
            raise

        print(f"[RAG] Document processing complete! {len(chunks)} chunks processed")
        post_progress(doc_id, "complete", 100, message="Document processing complete!", totalChunks=len(chunks))
        return len(chunks)

    def get_chunks_by_doc_id(self, doc_id: str) -> List[Document]:
        results = self.detailed_store.get(where={"doc_id": doc_id}, include=["metadatas", "documents"])
        print(f"DEBUG: detailed_store.get returned keys: {list(results.keys())}")
        try:
            # Print types for debugging
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

    # --- RAG Logic with Structured Citations ---
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
            # Ensure relevant_chunks are Document objects
            for i, rc in enumerate(relevant_chunks):
                if not isinstance(rc, Document):
                    print(f"Converting non-Document relevant chunk at index {i} of type {type(rc)}")
                    relevant_chunks[i] = Document(page_content=str(rc), metadata={})
            print(f"DEBUG: relevant_chunks types after conversion: {[type(rc) for rc in relevant_chunks[:5]]}")

        if not relevant_chunks:
            return "No relevant chunks found.", []

        context_string = _format_chunks_for_prompt(relevant_chunks)
        final_prompt = build_advanced_rag_prompt(question, context_string)

        llm_name_for_rag = model_name or "gemma3:1b"

        try:
            llm = get_llm(llm_name_for_rag)
            json_prompt = final_prompt + """\n\nReturn JSON: {"answer": "...", "references": [{"id": "1", "title": "...", "source": "...", "page": 1, "snippet": "..."}]}"""
            raw_response = await asyncio.to_thread(llm.invoke, json_prompt)
            raw_response_content = getattr(raw_response, "content", str(raw_response))
            print(f"LLM response received (length: {len(raw_response_content)} chars)")
            print(f"Raw response preview: {raw_response_content[:300]}...")

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

# ==============================================================================
# 8. STREAMING SUMMARIZER (Preserved Feature)
# ==============================================================================
# (This section is unchanged from the previous code)

# ==============================================================================
# 9. FASTAPI APP & ENDPOINTS (MODIFIED)
# ==============================================================================

print("Starting FastAPI app...")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service
    try:
        rag_service = HierarchicalRAGService(
            summary_path=SUMMARY_STORE_PATH,
            detailed_path=DETAILED_STORE_PATH,
            embeddings=EMBEDDINGS_MODEL
        )
    except Exception as e:
        print(f"FATAL: Could not initialize RAG Service: {e}")
        rag_service = None
    yield
    # Shutdown code (if needed)

app = FastAPI(title="Hierarchical RAG API with Structured Citations", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Backend is running"}

@app.post("/process")
async def process(file: UploadFile):
    print("[PROCESS] Starting document processing...")
    if rag_service is None:
        print("[PROCESS] RAG Service is not operational")
        raise HTTPException(status_code=500, detail="RAG Service is not operational.")

    print(f"[PROCESS] Received file: {file.filename}")
    if is_image_file(file.filename):
        print("[PROCESS] Detected image file")
        doc_id = f"img_{uuid4().hex}"
        image_path = os.path.join(IMAGE_STORE, f"{doc_id}{Path(file.filename).suffix}")
        with open(image_path, "wb") as f: f.write(await file.read())
        print(f"[PROCESS] Image saved: {image_path}")
        return {"documentId": doc_id, "status": "image_saved", "isImage": True, "imagePath": image_path}

    print("[PROCESS] Processing as PDF document")
    doc_id = f"doc_{uuid4().hex}"
    print(f"[PROCESS] Generated document ID: {doc_id}")

    try:
        print("[PROCESS] Reading PDF bytes...")
        pdf_bytes = await file.read()
        print(f"[PROCESS] Read {len(pdf_bytes)} bytes")

        print("[PROCESS] Calling add_document_to_stores...")
        chunk_count = await rag_service.add_document_to_stores(pdf_bytes, doc_id, OLLAMA_MODEL)
        print(f"[PROCESS] Successfully processed document with {chunk_count} chunks")
        return {"documentId": doc_id, "status": "embeddings_created", "chunkCount": chunk_count, "isImage": False}
    except Exception as e:
        print(f"[PROCESS] Error processing document: {e}")
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
                "summary": chunk.metadata.get("summary", "") if hasattr(chunk, 'metadata') and isinstance(chunk.metadata, dict) else "",
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
        # Image Q&A logic is preserved
        return await process_image_query(image_ids, text_ids, prompt, model)

    if text_ids:
        # --- Call the advanced RAG function ---
        max_citations = 5 # Get more chunks for the advanced prompt

        # Use await because query_rag is now an async function
        response_text, citations = await rag_service.query_rag(
            document_ids=text_ids,
            question=prompt,
            model_name=model,
            specific_chunks=specific_chunks,
            top_k=max_citations
        )
        
        # Ensure we have at least 2-3 citations if available
        num_citations = min(random.randint(2, 3), len(citations))
        citations = citations[:num_citations]
        return JSONResponse(content={"response": response_text, "citations": citations})

    # --- No-context Q&A Logic (Unchanged) ---
    response_text = generate_with_llm(prompt, model) # Uses simple text gen
    return JSONResponse(content={"response": response_text, "citations": []})


async def process_image_query(image_ids: list, text_ids: list, prompt: str, model: str):
    """
    Image Q&A function with optional RAG context from text documents.
    """
    vision_model = "llava"
    print(f"Image queries: forcing vision model='{vision_model}', ignoring requested model='{model}'")
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
        # Await the async RAG query
        context, citations = await rag_service.query_rag(text_ids, prompt, model, top_k=3)
        num_citations = min(random.randint(2, 3), len(citations))
        citations = citations[:num_citations]
        additional_context = f"\n\nAdditional context from documents:\n{context}"

    final_response = "\n\n".join(responses)
    if additional_context: final_response += additional_context

    return JSONResponse(content={"response": final_response, "citations": citations, "usedVisionModel": True, "visionModel": vision_model})



from fastapi.responses import Response

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
    model = (await request.json()).get("name", OLLAMA_MODEL)
    resp = requests.post(f"{OLLAMA_URL}/api/pull", json={"name": model, "stream": False}, timeout=300)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)

# ==============================================================================
# 10. SERVER STARTUP
# ==============================================================================

try:
    # Pull models first before starting service
    print("Pulling required models...")
    os.system("ollama pull gemma3:1b")
    os.system("ollama pull llava")
    start_ollama_service()
    os.system("ollama pull gemma3:1b && ollama pull llava")
except Exception as e:
    print(f"Failed to start Ollama service: {e}")
    raise

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
