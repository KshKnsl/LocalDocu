#!/usr/bin/env python3
"""
Private Document Summarizer - Backend for macOS
================================================
Complete privacy-focused document summarization using local Ollama models.

Prerequisites:
1. Install Ollama: https://ollama.ai/download (macOS installer)
2. Python 3.10+ (comes with macOS or use Homebrew: brew install python3)
3. Install dependencies: pip3 install -r requirements.txt

Setup:
1. Download this file and requirements.txt
2. Open Terminal in the same folder
3. Run: pip3 install -r requirements.txt
4. Set your Google API key (optional for remote model):
   export GOOGLE_API_KEY="your_api_key_here"
5. Make executable: chmod +x backend-mac.py
6. Run: python3 backend-mac.py
7. Copy the ngrok URL or use http://localhost:8000

GitHub: https://github.com/YourRepo/MinorProject
"""

import os
import subprocess
import threading
import time
import requests
import tempfile
import asyncio
import json
from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama.llms import OllamaLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from uuid import uuid4
import uvicorn
import sys

# Configuration
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
OLLAMA_URL = "http://localhost:11434"
PERSIST_BASE = os.path.abspath("./chroma_store")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

app = FastAPI(title="🦙 Private Document Summarizer API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def check_ollama_running():
    """Check if Ollama is running"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def start_ollama_mac():
    """Start Ollama service on macOS"""
    print("🦙 Starting Ollama service...")
    
    # Check if already running
    if check_ollama_running():
        print("✅ Ollama is already running!")
        return None
    
    # Try to start Ollama
    try:
        subprocess.Popen(["ollama", "serve"], 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL)
        
        # Wait for Ollama to start
        for _ in range(30):
            if check_ollama_running():
                print("✅ Ollama is running!")
                return True
            time.sleep(1)
        
        print("⚠️ Ollama didn't start automatically. Please start it manually.")
        return False
    except FileNotFoundError:
        print("❌ Ollama not found. Please install from https://ollama.ai/download")
        return False

def load_and_split_pdf(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        path = tmp.name
    docs = PyMuPDFLoader(path).load()
    os.remove(path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = splitter.split_documents(docs)
    print(f"📄 Loaded {len(chunks)} chunks.")
    return chunks

def create_persistent_vectorstore(chunks, persist_dir: str):
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    return vs

def generate_with_llm(prompt: str, model_name: str):
    """Unified LLM generation for both Gemini and Ollama"""
    if model_name.lower() == "remote":
        if not GOOGLE_API_KEY:
            return "Error: GOOGLE_API_KEY not set. Please set environment variable."
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GOOGLE_API_KEY)
        resp = llm.invoke(prompt)
        return getattr(resp, "content", str(resp))
    else:
        ollama = OllamaLLM(model=model_name)
        resp = ollama.invoke(prompt)
        return getattr(resp, "content", str(resp))

def build_rag_prompt(context: str, question: str) -> str:
    return (
        f"Answer the following question based on the provided context. Be specific and detailed in your answer.\n\n"
        f"Context:\n{context[:20000]}\n\nQuestion: {question}\n\nAnswer:"
    )

def build_context_from_documents(document_ids, question: str, top_k: int = 5):
    """Load persisted Chroma stores and return context + citations"""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    contexts, citations, seen = [], [], set()
    
    for doc_id in document_ids:
        persist_dir = os.path.join(PERSIST_BASE, doc_id)
        vs = Chroma(embedding_function=embeddings, persist_directory=persist_dir)
        docs = vs.similarity_search(question, k=top_k)
        for idx, d in enumerate(docs):
            snippet = d.page_content.strip()
            if snippet not in seen:
                contexts.append(snippet)
                citations.append({
                    "documentId": doc_id,
                    "page": d.metadata.get("page", d.metadata.get("page_number", "N/A")),
                    "snippet": snippet[:300],
                    "fullText": snippet,
                    "source": d.metadata.get("source", doc_id),
                    "rank": idx + 1
                })
                seen.add(snippet)
    
    return "\n---\n".join(contexts)[:20000], citations

# API Endpoints
@app.get("/")
def home():
    return {"message": "Private Document Summarizer API - macOS", "status": "active"}

@app.post("/process")
async def process(file: UploadFile):
    chunks = load_and_split_pdf(await file.read())
    doc_id = f"doc_{uuid4().hex}"
    create_persistent_vectorstore(chunks, os.path.join(PERSIST_BASE, doc_id))
    return {"documentId": doc_id, "status": "embeddings_created", "chunkCount": len(chunks)}

@app.post("/summarize_by_id")
async def summarize_by_id(request: Request):
    data = await request.json()
    document_id = data.get("documentId")
    model_name = data.get("model_name", OLLAMA_MODEL)

    persist_dir = os.path.join(PERSIST_BASE, document_id)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = Chroma(embedding_function=embeddings, persist_directory=persist_dir)
    docs = vs.get().get("documents", [])

    def build_chunk_summary_prompt(chunk_text: str, idx: int, total: int, max_words: int = 300) -> str:
        txt = (chunk_text or "")[:30000]
        return (
            f"You are an expert research assistant. Summarize the following document CHUNK ({idx+1}/{total}) in up to {max_words} words.\n"
            "Focus on the most important claims, methods, and results in this chunk. If this chunk contains tables or figures, summarize their findings briefly. Keep the output as a short, self-contained bullet list or short paragraphs that highlight key points.\n\n"
            f"Chunk content:\n{txt}\n\nIntermediate summary:\n"
        )

    def build_synthesis_prompt(intermediate_summaries: list[str], max_words: int = 500) -> str:
        combined = "\n\n".join(intermediate_summaries)[:40000]
        return (
            f"You are an expert research assistant. You have the following INTERMEDIATE SUMMARIES derived from chunks of a research paper.\n"
            f"Using only the information below, produce a single structured summary of approximately {max_words} words with the following sections:\n"
            "- TL;DR (one short paragraph)\n- Problem & Motivation\n- Method\n- Key Results (with concise evidence/metrics if present)\n- Limitations\n- Conclusion\n\n"
            "Write clearly and use short headings for each section. If information is missing, be explicit (e.g., 'Not enough information provided').\n\n"
            f"Intermediate summaries:\n{combined}\n\nFinal structured summary:\n"
        )

    semaphore = asyncio.Semaphore(10)

    async def summarize_chunk_async(chunk_text: str, idx: int, total: int) -> str:
        prompt = build_chunk_summary_prompt(chunk_text, idx, total)
        async with semaphore:
            return await asyncio.to_thread(generate_with_llm, prompt, model_name)
    
    async def stream_summaries():
        tasks = [asyncio.create_task(summarize_chunk_async(c, i, len(docs))) for i, c in enumerate(docs)]
        intermediate_summaries = []
        
        for i, task in enumerate(asyncio.as_completed(tasks)):
            summary = await task
            intermediate_summaries.append(summary)
            yield json.dumps({"type": "chunk", "index": i, "total": len(docs), "summary": summary}) + "\n"
        
        yield json.dumps({"type": "status", "message": "Generating final summary..."}) + "\n"
        synthesis_prompt = build_synthesis_prompt(intermediate_summaries)
        final_summary = await asyncio.to_thread(generate_with_llm, synthesis_prompt, model_name)
        
        yield json.dumps({
            "type": "final",
            "summary": final_summary,
            "chunkCount": len(docs),
            "intermediateCount": len(intermediate_summaries)
        }) + "\n"
    
    return StreamingResponse(stream_summaries(), media_type="application/x-ndjson")

@app.post("/pull")
async def pull_model(request: Request):
    model = (await request.json()).get("name", OLLAMA_MODEL)
    if model.lower() == "remote":
        return {"message": "Done"}
    resp = requests.post(f"{OLLAMA_URL}/api/pull", json={"name": model, "stream": False})
    return JSONResponse(content=resp.json(), status_code=resp.status_code)

@app.post("/generate")
async def generate_text(request: Request):
    data = await request.json()
    model = data.get("model", OLLAMA_MODEL)
    prompt = data.get("prompt", "")
    document_ids = data.get("documentIds", [])
    use_mcp_tools = data.get("use_agent_tools", False)
    lower = prompt.lower()
    max_citations = 5 if any(k in lower for k in ["cite", "source", "references", "reference", "where did", "which paper", "according to"]) else 3

    print(f"🎯 /generate → model={model}, mcp_tools={use_mcp_tools}, docs={len(document_ids)}")
    citations = []
    
    if use_mcp_tools:
        return JSONResponse(content={
            "response": "Plagiarism checker is not yet implemented.",
            "citations": [],
            "toolCalls": [],
            "status": "not_implemented"
        })
    
    if document_ids and len(document_ids) > 0:
        context, citations = build_context_from_documents(document_ids, prompt, top_k=max_citations)
        prompt = build_rag_prompt(context, prompt)

    response_text = generate_with_llm(prompt, model)
    return JSONResponse(content={"response": response_text, "citations": citations})

if __name__ == "__main__":
    print("=" * 60)
    print("🔒 Private Document Summarizer - macOS Backend")
    print("=" * 60)
    
    # Start Ollama
    start_ollama_mac()
    
    # Start ngrok tunnel (optional)
    use_ngrok = input("\n🌐 Do you want to expose via ngrok? (y/n): ").lower() == 'y'
    
    if use_ngrok:
        ngrok_token = input("Enter your ngrok auth token (from https://dashboard.ngrok.com): ")
        if ngrok_token:
            ngrok.set_auth_token(ngrok_token)
            public_url = ngrok.connect(8000)
            print(f"\n✅ Public URL: {public_url}")
            print(f"📋 Copy this URL and paste it in the frontend Backend Settings\n")
    
    print("\n🚀 Starting FastAPI server...")
    print("📍 Local URL: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
