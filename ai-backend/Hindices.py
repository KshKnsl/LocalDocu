# ==============================================================================
# 0. INSTALLS (Colab only - comment out for local use)
# ==============================================================================
# Uncomment the following lines if running in Google Colab:
# !curl -fsSL https://ollama.com/install.sh | sh
# !pip install fastapi uvicorn pyngrok requests boto3 python-multipart aiofiles langchain langchain-community chromadb sentence-transformers PyMuPDF langchain-huggingface langchain-chroma langchain-google-genai langchain-ollama langchain-experimental flashrank pydantic python-dotenv

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import os, signal, psutil, gc, time, sys, subprocess, threading, requests, tempfile, asyncio, json, base64
from pathlib import Path
from uuid import uuid4
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv() 

# --- FastAPI & Server ---
from fastapi import FastAPI, UploadFile, Form, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from pyngrok import ngrok

# --- LangChain Core ---
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

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
        GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
        NGROK_AUTHTOKEN = userdata.get("NGROK_AUTHTOKEN")
    except Exception:
        print("WARNING: Could not load from Colab secrets, falling back to environment variables.")
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
        NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN")
else:
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN")

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY":
    print("WARNING: GOOGLE_API_KEY not configured properly. Set it in .env file.")
if not NGROK_AUTHTOKEN or NGROK_AUTHTOKEN == "YOUR_NGROK_AUTHTOKEN":
    print("WARNING: NGROK_AUTHTOKEN not configured properly. Set it in .env file.")

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
OLLAMA_URL = "http://localhost:11434"

PROXY_BACKEND_URL = os.environ.get("PROXY_BACKEND_URL", "http://localhost:3000")

def post_progress(document_id: str, status: str, progress: int = 0, **kwargs):
    """Post progress update to proxy backend."""
    try:
        payload = {
            "documentId": document_id,
            "status": status,
            "progress": progress,
            **kwargs
        }
        requests.post(f"{PROXY_BACKEND_URL}/api/progress", json=payload, timeout=2)
    except Exception as e:
        print(f"Warning: Failed to post progress: {e}")

# --- Persistent Storage Paths (Hierarchical) ---
PERSIST_BASE = os.path.abspath("./chroma_store")
SUMMARY_STORE_PATH = os.path.join(PERSIST_BASE, "summary_store")
DETAILED_STORE_PATH = os.path.join(PERSIST_BASE, "detailed_store")
IMAGE_STORE = os.path.abspath("./image_store")

os.makedirs(SUMMARY_STORE_PATH, exist_ok=True)
os.makedirs(DETAILED_STORE_PATH, exist_ok=True)
os.makedirs(IMAGE_STORE, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# --- Global Reusable Components ---
EMBEDDINGS_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
GLOBAL_RERANKER = FlashrankRerank(top_n=5) # Default re-ranker

# ==============================================================================
# 3. NEW: Pydantic Models for Structured Output
# ==============================================================================

class Reference(BaseModel):
    """Pydantic model for a single citation reference."""
    id: str = Field(..., description="The citation ID, e.g., '1', '2'.")
    title: str = Field(..., description="The title of the source document.")
    source: str = Field(..., description="The source URL or filename.")

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
        if ref.source not in unique_refs:
            new_id = str(new_id_counter)
            unique_refs[ref.source] = Reference(id=new_id, title=ref.title, source=ref.source)
            new_id_counter += 1
        
        id_mapping[ref.id] = unique_refs[ref.source].id

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
# (This section is unchanged from the previous code)

def kill_processes():
    print("Terminating existing processes...")
    os.system("pkill -f 'uvicorn' || true")
    os.system("pkill -f 'ngrok' || true")
    os.system("pkill -f 'ollama' || true")
    gc.collect()
    time.sleep(2)
    print("✅ All background processes and threads terminated.")

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
    print("🦙 Starting Ollama service...")
    for _ in range(40):
        try:
            if requests.get(OLLAMA_URL).status_code == 200:
                print("✅ Ollama is running locally!\n")
                return True
        except:
            time.sleep(2)
    raise RuntimeError("❌ Ollama failed to start in time.")

def is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS

# ==============================================================================
# 6. SHARED LLM & RAG PROMPT LOGIC (MODIFIED)
# ==============================================================================

def get_llm(model_name: str):
    """Unified function to get an LLM instance."""
    if model_name.lower() == "remote":
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GOOGLE_API_KEY)
    else:
        # Using a model known to be good at JSON mode
        if "json" not in model_name:
             print(f"Warning: Using '{model_name}'. For best structured output, try 'mistral:latest' or 'llama3:latest' with format=json.")
        
        return OllamaLLM(model=model_name, format="json", temperature=0)

def generate_with_llm(prompt: str, model_name: str):
    """Unified function to invoke an LLM for *non-structured* text."""
    # Use a basic model for simple generation
    if model_name.lower() == "remote":
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GOOGLE_API_KEY)
    else:
        llm = OllamaLLM(model=model_name, temperature=0.1)
        
    resp = llm.invoke(prompt)
    return getattr(resp, "content", str(resp))

def _format_chunks_for_prompt(chunks: List[Document]) -> str:
    """Formats retrieved chunks into the string format"""
    context_strings = []
    for i, chunk in enumerate(chunks):
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
      "source": "https://example.com/ai.pdf"
    }},
    {{
      "id": "2",
      "title": "ML Basics (Page 2, Chunk 4)",
      "source": "https://example.com/ml.pdf"
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

# ==============================================================================
# 7. CORE: HIERARCHICAL RAG SERVICE (MODIFIED)
# ==============================================================================

class HierarchicalRAGService:
    """
    Manages the Hierarchical Vector Stores (Summary & Detailed)
    and all core RAG logic.
    """
    def __init__(self, summary_path, detailed_path, embeddings):
        print("Initializing HierarchicalRAGService...")
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
        print(f"✅ Summary Store loaded: {self.summary_store._collection.count()} items")
        print(f"✅ Detailed Store loaded: {self.detailed_store._collection.count()} items")

    # --- Ingestion Logic (Unchanged) ---
    def _load_and_split_pdf(self, pdf_bytes: bytes) -> List[Document]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            path = tmp.name
        docs = PyMuPDFLoader(path).load()
        os.remove(path)
        if not docs: return []
        print("🧠 Applying Semantic Chunker...")
        splitter = SemanticChunker(self.embeddings, breakpoint_threshold_type="percentile")
        chunks = splitter.split_documents(docs)
        print(f"📄 Loaded {len(chunks)} semantic chunks.")
        return chunks

    async def _generate_summary_for_ingestion(self, chunks: List[Document], model_name: str) -> str:
        if not chunks: return "No text content found."
        print(f"Generating ingestion summary for {len(chunks)} chunks...")
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
        print("✅ Ingestion summary generated.")
        return final_summary

    async def add_document_to_stores(self, pdf_bytes: bytes, doc_id: str, model_name: str):
        post_progress(doc_id, "loading", 5, message="Loading PDF...")
        
        chunks = self._load_and_split_pdf(pdf_bytes)
        if not chunks:
            post_progress(doc_id, "failed", 0, message="No text content found")
            raise ValueError("No text content found in the document")
        
        post_progress(doc_id, "chunking", 20, message=f"Split into {len(chunks)} chunks", totalChunks=len(chunks))
        
        # Associate original filename with all chunks for better citation
        # Let's assume the first chunk has the source metadata
        source_filename = chunks[0].metadata.get("source", f"doc_{doc_id}")

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
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["title"] = f"{Path(source_filename).name} (Page {chunk.metadata.get('page', i+1)})"
            if i % 5 == 0 or i == len(chunks) - 1: 
                progress = 75 + int((i + 1) / len(chunks) * 20)
                post_progress(doc_id, "embedding_chunks", progress, 
                            message=f"Embedding chunk {i+1}/{len(chunks)}...",
                            currentChunk=i+1, totalChunks=len(chunks))
        
        chunk_ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        self.detailed_store.add_documents(chunks, ids=chunk_ids)

        post_progress(doc_id, "complete", 100, message="Document processing complete!", totalChunks=len(chunks))
        print(f"✅ Document {doc_id} added to hierarchical stores.")
        return len(chunks)

    def get_chunks_by_doc_id(self, doc_id: str) -> List[Document]:
        results = self.detailed_store.get(where={"doc_id": doc_id}, include=["metadatas", "documents"])
        if not results.get('documents'): return []
        docs = [Document(page_content=text, metadata=results['metadatas'][i]) for i, text in enumerate(results['documents'])]
        print(f"Found {len(docs)} chunks for doc_id {doc_id}")
        return docs

    # --- RAG Logic with Structured Citations ---
    async def query_rag(self, document_ids: List[str], question: str, model_name: str, top_k: int = 5) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Performs the 2-step Hierarchical RAG query using
        structured output and IEEE-style citations.
        """
        print(f"\n{'='*80}")
        print(f"🔍 Starting Hierarchical RAG query")
        print(f"{'='*80}")
        print(f"📋 Question: {question}")
        print(f"📁 Document IDs: {document_ids}")
        print(f"🤖 Model: {model_name}")
        print(f"🎯 Top K: {top_k}")
        
        # === Step 1: Search SUMMARY_STORE (Unchanged) ===
        summary_retriever = self.summary_store.as_retriever(
            search_kwargs={'k': 20, 'filter': {'doc_id': {'$in': document_ids}}}
        )
        summary_compressor = ContextualCompressionRetriever(
            base_compressor=GLOBAL_RERANKER, base_retriever=summary_retriever
        )
        print("... (Step 1) Re-ranking summaries...")
        relevant_summaries = summary_compressor.invoke(question)
        print(f"✓ Found {len(relevant_summaries)} relevant summaries")
        relevant_doc_ids = list(set([doc.metadata['doc_id'] for doc in relevant_summaries]))
        
        if not relevant_doc_ids:
            print("No relevant documents found in summary search.")
            return "I couldn't find any relevant documents for your question.", []
        print(f"... (Step 1) Found top relevant docs: {relevant_doc_ids}")

        # === Step 2: Search DETAILED_STORE (Unchanged) ===
        detailed_retriever = self.detailed_store.as_retriever(
            search_kwargs={'k': 25, 'filter': {'doc_id': {'$in': relevant_doc_ids}}}
        )
        chunk_compressor = ContextualCompressionRetriever(
            base_compressor=FlashrankRerank(top_n=top_k), 
            base_retriever=detailed_retriever
        )
        print("... (Step 2) Re-ranking detailed chunks...")
        relevant_chunks = chunk_compressor.invoke(question)
        print(f"✓ Found {len(relevant_chunks)} relevant chunks after reranking")
        
        if not relevant_chunks:
            print("No relevant chunks found in detailed search.")
            return "I found relevant documents, but no specific chunks matched your question.", []
        
        # Print chunk details
        print(f"\n📝 Chunk Details:")
        for i, chunk in enumerate(relevant_chunks[:3], 1):  # Show first 3
            print(f"  Chunk {i}:")
            print(f"    - Title: {chunk.metadata.get('title', 'N/A')}")
            print(f"    - Source: {chunk.metadata.get('source', 'N/A')}")
            print(f"    - Page: {chunk.metadata.get('page', 'N/A')}")
            print(f"    - Content preview: {chunk.page_content[:100]}...")

        # === Step 3: Build Advanced Prompt & Call Structured LLM ===
        print("... (Step 3) Building advanced RAG prompt...")
        context_string = _format_chunks_for_prompt(relevant_chunks)
        final_prompt = build_advanced_rag_prompt(question, context_string)
        
        # Get the LLM *with* structured output
        print(f"... (Step 3) Calling model '{model_name}' for structured JSON output...")
        
        # We must use a model that supports JSON mode well, like mistral, llama3, or gpt/gemini
        llm_name_for_rag = model_name
        if model_name.lower() != "remote" and model_name.lower() not in ["mistral", "llama3"]:
            print(f"Switching RAG model from '{model_name}' to 'mistral' for better JSON support.")
            llm_name_for_rag = "mistral"
        
        print(f"🤖 Using model: '{llm_name_for_rag}' for structured output")

        try:
            llm = get_llm(llm_name_for_rag)
            is_remote_model = llm_name_for_rag.lower() == "remote"
            
            if is_remote_model:
                print("... Creating structured LLM instance (Google Gemini)...")
                structured_llm = llm.with_structured_output(AIAnswer)
                print(f"✓ Structured LLM created: {type(structured_llm)}")
                ai_answer_response = await asyncio.to_thread(structured_llm.invoke, final_prompt)
                print(f"✓ LLM response received: {type(ai_answer_response)}")
                
            else:
                print("... Using JSON mode for Ollama model (no native structured output)...")
                json_prompt = final_prompt + """\n\nIMPORTANT: Return your response as a valid JSON object with this exact structure:
{
  "answer": "your detailed answer here with [1], [2] citation markers",
  "references": [
    {"id": 1, "title": "document title", "source": "source path", "page": page_number},
    {"id": 2, "title": "document title", "source": "source path", "page": page_number}
  ]
}

Ensure all citation markers in your answer correspond to reference IDs."""
                
                print("... Invoking Ollama LLM with JSON instructions...")
                raw_response = await asyncio.to_thread(llm.invoke, json_prompt)
                print(f"✓ LLM response received (length: {len(raw_response.content)} chars)")
                print(f"📄 Raw response preview: {raw_response.content[:300]}...")
                
                import json
                import re
                
                response_text = raw_response.content
                
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    print("✓ Found JSON in code block")
                else:
                    json_match = re.search(r'\{.*"answer".*"references".*\}', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        print("✓ Found raw JSON object")
                    else:
                        print("❌ No JSON structure found in response")
                        raise ValueError("No JSON structure in response")
                
                parsed_json = json.loads(json_str)
                print(f"✓ JSON parsed successfully")
                
                references = [
                    Reference(
                        id=ref.get('id', i+1),
                        title=ref.get('title', 'Unknown'),
                        source=ref.get('source', 'Unknown'),
                        page=ref.get('page', 0)
                    )
                    for i, ref in enumerate(parsed_json.get('references', []))
                ]
                
                ai_answer_response = AIAnswer(
                    answer=parsed_json.get('answer', ''),
                    references=references
                )
                print(f"✓ Converted to AIAnswer object")
            
            print(f"✓ Answer length: {len(ai_answer_response.answer)} chars")
            print(f"✓ References count: {len(ai_answer_response.references)}")
            print(f"📝 Answer preview: {ai_answer_response.answer[:200]}...")
            
            print(f"\n{'─'*80}")
            print(f"🔄 STEP 4: Post-processing")
            print(f"{'─'*80}")
            final_answer, final_refs = deduplicate_references_and_update_answer(
                ai_answer_response.answer, 
                ai_answer_response.references
            )
            
            # Convert Pydantic models to dicts for JSON response
            final_refs_dict = [ref.model_dump() for ref in final_refs]
            
            return final_answer, final_refs_dict
            
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"❌ CRITICAL ERROR in Structured Output")
            print(f"{'='*80}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {e}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            print(f"\n{'─'*80}")
            print(f"🔄 Failing over to simple text generation...")
            print(f"{'─'*80}")
            # Failover to simple text generation
            print("... Building simple prompt (no citations)...")
            simple_context = "\n\n".join([c.page_content for c in relevant_chunks])
            simple_prompt = f"Answer this question based on the context:\nQuestion: {question}\n\nContext:\n{simple_context}\n\nAnswer:"
            print(f"📏 Simple prompt length: {len(simple_prompt)} characters")
            print("... Calling LLM without structured output...")
            failover_answer = generate_with_llm(simple_prompt, model_name)
            print(f"✓ Failover answer received: {len(failover_answer)} chars")
            print(f"\n{'='*80}")
            print(f"⚠️  RAG Complete (Failover Mode - No Citations)")
            print(f"{'='*80}")
            return failover_answer, []

# ==============================================================================
# 8. STREAMING SUMMARIZER (Preserved Feature)
# ==============================================================================
# (This section is unchanged from the previous code)

async def stream_map_reduce_summary(docs_text: List[str], model_name: str):
    def build_chunk_summary_prompt(chunk_text: str, idx: int, total: int, max_words: int = 300) -> str:
        txt = (chunk_text or "")[:30000]
        return (
            f"You are an expert research assistant. Summarize the following document CHUNK ({idx+1}/{total}) in up to {max_words} words.\n"
            "Focus on the most important claims, methods, and results in this chunk. Keep the output as a short, self-contained bullet list or short paragraphs.\n\n"
            f"Chunk content:\n{txt}\n\nIntermediate summary:\n"
        )
    def build_synthesis_prompt(intermediate_summaries: list[str], max_words: int = 500) -> str:
        combined = "\n\n".join(intermediate_summaries)[:40000]
        return (
            f"You are an expert research assistant. You have the following INTERMEDIATE SUMMARIES derived from chunks of a research paper.\n"
            f"Using only the information below, produce a single structured summary of approximately {max_words} words with the following sections:\n"
            "- TL;DR (one short paragraph)\n- Problem & Motivation\n- Method\n- Key Results\n- Limitations\n- Conclusion\n\n"
            "Write clearly and use short headings for each section. If information is missing, be explicit.\n\n"
            f"Intermediate summaries:\n{combined}\n\nFinal structured summary:\n"
        )
    semaphore = asyncio.Semaphore(10)
    async def summarize_chunk_async(chunk_text: str, idx: int, total: int) -> str:
        prompt = build_chunk_summary_prompt(chunk_text, idx, total)
        async with semaphore:
            return await asyncio.to_thread(generate_with_llm, prompt, model_name)

    total_chunks = len(docs_text)
    tasks = [asyncio.create_task(summarize_chunk_async(c, i, total_chunks)) for i, c in enumerate(docs_text)]
    intermediate_summaries = []
    
    for i, task in enumerate(asyncio.as_completed(tasks)):
        try:
            summary = await task
            intermediate_summaries.append(summary)
            yield json.dumps({"type": "chunk", "index": i, "total": total_chunks, "summary": summary}) + "\n"
        except Exception as e:
            print(f"Error summarizing chunk {i}: {e}")
            yield json.dumps({"type": "error", "index": i, "message": str(e)}) + "\n"
    yield json.dumps({"type": "status", "message": "Generating final summary..."}) + "\n"
    synthesis_prompt = build_synthesis_prompt(intermediate_summaries)
    final_summary = await asyncio.to_thread(generate_with_llm, synthesis_prompt, model_name)
    yield json.dumps({
        "type": "final",
        "summary": final_summary,
        "chunkCount": total_chunks,
        "intermediateCount": len(intermediate_summaries)
    }) + "\n"

# ==============================================================================
# 9. FASTAPI APP & ENDPOINTS (MODIFIED)
# ==============================================================================

print("Starting FastAPI app...")
app = FastAPI(title="🦙 Hierarchical RAG API with Structured Citations")

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
        print("RAG service failed to initialize. Endpoints will be disabled.")

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
        print(f"🖼️ Image saved: {image_path}")
        return {"documentId": doc_id, "status": "image_saved", "isImage": True, "imagePath": image_path}
    
    doc_id = f"doc_{uuid4().hex}"
    try:
        pdf_bytes = await file.read()
        chunk_count = await rag_service.add_document_to_stores(pdf_bytes, doc_id, "mistral") # Use fast model for ingestion
        return {"documentId": doc_id, "status": "embeddings_created", "chunkCount": chunk_count, "isImage": False}
    except Exception as e:
        print(f"Error processing document: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to process document", "message": str(e)})

@app.post("/summarize_by_id")
async def summarize_by_id(request: Request):
    if rag_service is None: raise HTTPException(status_code=500, detail="RAG Service is not operational.")
    data = await request.json()
    document_id = data.get("documentId")
    model_name = data.get("model_name", OLLAMA_MODEL)
    if not document_id: raise HTTPException(status_code=400, detail="documentId is required")
    chunks = rag_service.get_chunks_by_doc_id(document_id)
    if not chunks: raise HTTPException(status_code=404, detail=f"No chunks found for documentId {document_id}")
    chunks_text = [c.page_content for c in chunks]
    return StreamingResponse(stream_map_reduce_summary(chunks_text, model_name), media_type="application/x-ndjson")

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
    
    image_ids = [doc_id for doc_id in document_ids if doc_id.startswith("img_")]
    text_ids = [doc_id for doc_id in document_ids if doc_id.startswith("doc_")]
    
    if image_ids:
        # Image Q&A logic is preserved
        return await process_image_query(image_ids, text_ids, prompt, model)
    
    if text_ids:
        # --- Call the advanced RAG function ---
        print(f"🎯 /generate → Hierarchical RAG query")
        max_citations = 7 # Get more chunks for the advanced prompt
        
        # Use await because query_rag is now an async function
        response_text, citations = await rag_service.query_rag(
            document_ids=text_ids,
            question=prompt,
            model_name=model,
            top_k=max_citations
        )
        return JSONResponse(content={"response": response_text, "citations": citations})

    # --- No-context Q&A Logic (Unchanged) ---
    print("🎯 /generate → No context (direct to LLM)")
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
        # Await the async RAG query
        context, citations = await rag_service.query_rag(text_ids, prompt, model, top_k=3)
        additional_context = f"\n\nAdditional context from documents:\n{context}"
    
    final_response = "\n\n".join(responses)
    if additional_context: final_response += additional_context
    
    return JSONResponse(content={"response": final_response, "citations": citations, "usedVisionModel": True, "visionModel": vision_model})

@app.post("/pull")
async def pull_model(request: Request):
    try:
        model = (await request.json()).get("name", OLLAMA_MODEL)
        if model.lower() == "remote":
            return {"message": "Using remote model, no pull needed."}
        print(f"Pulling model: {model}...")
        resp = requests.post(f"{OLLAMA_URL}/api/pull", json={"name": model, "stream": False}, timeout=300)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ==============================================================================
# 10. SERVER STARTUP
# ==============================================================================

kill_processes()

try:
    start_ollama_service()
    print("Pre-pulling 'mistral' for ingestion/RAG and 'llava' for vision...")
    os.system(f"ollama pull mistral")
    os.system("ollama pull llava")
    print("✅ Default models pulled.")
except Exception as e:
    print(f"Failed to start Ollama: {e}")

FIXED_URL = "https://mari-unbequeathed-milkily.ngrok-free.app"
public_url = "http://localhost:8000"
ngrok_enabled = False
ngrok_proc = None

if NGROK_AUTHTOKEN and NGROK_AUTHTOKEN != "YOUR_NGROK_AUTHTOKEN":
    if IN_COLAB:
        get_ipython().system(f'ngrok config add-authtoken {NGROK_AUTHTOKEN}')
    else:
        os.system(f"ngrok config add-authtoken {NGROK_AUTHTOKEN}")
    
    print("🌐 Starting ngrok tunnel with fixed URL...")
    ngrok_proc = subprocess.Popen(
        ["ngrok", "http", "--host-header=rewrite", "--log=stdout", "--url", FIXED_URL, "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    public_url = FIXED_URL
    ngrok_enabled = True
    
    time.sleep(3)
    print(f"✅ Public URL: {public_url}\n")
else:
    print("⚠️ Ngrok not configured. Server will only be accessible locally.")

config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)
server = uvicorn.Server(config)

def run_uvicorn():
    print("🚀 FastAPI server starting on port 8000...")
    asyncio.run(server.serve())

threading.Thread(target=run_uvicorn, daemon=True).start()

print("\n🎉 Your API is live! 🎉")
print(f"Access it at: {public_url}")

try:
    while True: time.sleep(300)
except KeyboardInterrupt:
    print("Shutting down server...")
    if ngrok_enabled and ngrok_proc:
        ngrok_proc.terminate()
        ngrok_proc.wait()
    print("Goodbye!")