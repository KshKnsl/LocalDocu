!curl -fsSL https://ollama.com/install.sh | sh
!pip install fastapi uvicorn pyngrok requests boto3 python-multipart aiofiles langchain langchain-community chromadb sentence-transformers PyMuPDF langchain-huggingface langchain-chroma langchain-google-genai langchain-ollama
import os, signal, psutil, gc, time

os.system("pkill -f 'uvicorn' || true")
os.system("pkill -f 'ngrok' || true")
os.system("pkill -f 'ollama' || true")
gc.collect()
time.sleep(2)
print("✅ All background processes and threads terminated.")

import os, subprocess, threading, time, requests, tempfile, asyncio
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from pyngrok import ngrok
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama.llms import OllamaLLM
from uuid import uuid4
import uvicorn, sys
from google.colab import userdata
GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
from langchain_google_genai import ChatGoogleGenerativeAI

def stream_logs(proc, name):
    for line in iter(proc.stdout.readline, b''):
        sys.stdout.write(f"[{name}] {line.decode()}")
        sys.stdout.flush()
    for line in iter(proc.stderr.readline, b''):
        sys.stdout.write(f"[{name}-ERR] {line.decode()}")
        sys.stdout.flush()

ollama_proc = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
threading.Thread(target=stream_logs, args=(ollama_proc, "Ollama"), daemon=True).start()
print("🦙 Starting Ollama service...")

for _ in range(40):
    try:
        if requests.get("http://localhost:11434").status_code == 200:
            print("✅ Ollama is running locally!\n")
            break
    except:
        time.sleep(2)
else:
    raise RuntimeError("❌ Ollama failed to start in time.")

# === ⚙️ FASTAPI SETUP ===
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral") # Get model from environment variable, default to mistral
OLLAMA_URL = "http://localhost:11434"
PERSIST_BASE = os.path.abspath("./chroma_store")
app = FastAPI(title="🦙 Ollama + LangChain Summarizer API")

def load_and_split_pdf(pdf_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        path = tmp.name
    docs = PyMuPDFLoader(path).load()
    os.remove(path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"📄 Loaded {len(chunks)} chunks.")
    return chunks


def create_persistent_vectorstore(chunks, persist_dir: str):
    os.makedirs(persist_dir, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    return vs


# ===============================
# Utility Functions
# ===============================
def generate_with_llm(prompt: str, model_name: str):
    """Unified LLM generation for both Gemini and Ollama"""
    if model_name.lower() == "remote":
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        resp = llm.invoke(prompt)
        return getattr(resp, "content", str(resp))
    else:
        ollama = OllamaLLM(model=model_name)
        resp = ollama.invoke(prompt)
        return getattr(resp, "content", str(resp))

def build_rag_prompt(context: str, question: str) -> str:
    """Build RAG prompt with context"""
    return (
        f"Answer using ONLY this context. If not found, say 'I don't know'.\n\n"
        f"Context:\n{context[:20000]}\n\nQuestion: {question}\n\nAnswer:"
    )

def build_context_from_documents(document_ids, question: str, top_k: int = 5):
    """Load persisted Chroma stores and return context + citations"""
    if not document_ids:
        return "", []
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    contexts, citations, seen = [], [], set()
    
    for doc_id in document_ids:
        persist_dir = os.path.join(PERSIST_BASE, doc_id)
        if not os.path.isdir(persist_dir):
            continue
        vs = Chroma(embedding_function=embeddings, persist_directory=persist_dir)
        docs = vs.similarity_search(question, k=top_k)
        for idx, d in enumerate(docs):
            snippet = (d.page_content or "").strip()
            if snippet and snippet not in seen:
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

# ===============================
# MCP Tools Configuration
# ===============================
def get_document_content(document_ids: list[str], query: str = "") -> str:
    """Get content from uploaded research documents"""
    if not document_ids:
        return "No documents loaded."
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    texts = []
    for doc_id in document_ids:
        persist_dir = os.path.join(PERSIST_BASE, doc_id)
        if os.path.isdir(persist_dir):
            vs = Chroma(embedding_function=embeddings, persist_directory=persist_dir)
            docs = vs.get().get("documents", [])
            texts.extend(docs)
    return f"Documents content:\n{' '.join(texts)[:20000]}"


# ===============================
# API Endpoints
# ===============================
@app.get("/")
def home():
    return {"message": "✅ Ollama / Gemini + LangChain API active"}


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

    if not document_id:
        return JSONResponse(content={"error": "documentId required"}, status_code=400)

    persist_dir = os.path.join(PERSIST_BASE, document_id)
    if not os.path.isdir(persist_dir):
        return JSONResponse(content={"error": f"Document '{document_id}' not found"}, status_code=404)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = Chroma(embedding_function=embeddings, persist_directory=persist_dir)
    docs = vs.get().get("documents", [])

    if not docs:
        return {"summary": "No text found"}

    # Helper: build per-chunk summary prompt
    def build_chunk_summary_prompt(chunk_text: str, idx: int, total: int, max_words: int = 150) -> str:
        txt = (chunk_text or "")[:20000]
        return (
            f"You are an expert research assistant. Summarize the following document CHUNK ({idx+1}/{total}) in up to {max_words} words.\n"
            "Focus on the most important claims, methods, and results in this chunk. If this chunk contains tables or figures, summarize their findings briefly. Keep the output as a short, self-contained bullet list or short paragraphs that highlight key points.\n\n"
            f"Chunk content:\n{txt}\n\nIntermediate summary:\n"
        )

    # Helper: build final synthesis prompt
    def build_synthesis_prompt(intermediate_summaries: list[str], max_words: int = 250) -> str:
        combined = "\n\n".join(intermediate_summaries)[:20000]
        return (
            f"You are an expert research assistant. You have the following INTERMEDIATE SUMMARIES derived from chunks of a research paper.\n"
            f"Using only the information below, produce a single structured summary of approximately {max_words} words with the following sections:\n"
            "- TL;DR (one short paragraph)\n- Problem & Motivation\n- Method\n- Key Results (with concise evidence/metrics if present)\n- Limitations\n- Conclusion\n\n"
            "Write clearly and use short headings for each section. If information is missing, be explicit (e.g., 'Not enough information provided').\n\n"
            f"Intermediate summaries:\n{combined}\n\nFinal structured summary:\n"
        )

    # Limit concurrency to avoid overloading local model/service
    semaphore = asyncio.Semaphore(6)

    async def summarize_chunk_async(chunk_text: str, idx: int, total: int) -> str:
        prompt = build_chunk_summary_prompt(chunk_text, idx, total)
        async with semaphore:
            return await asyncio.to_thread(generate_with_llm, prompt, model_name)

    tasks = [asyncio.create_task(summarize_chunk_async(c, i, len(docs))) for i, c in enumerate(docs)]
    intermediate_summaries = await asyncio.gather(*tasks)
    synthesis_prompt = build_synthesis_prompt(intermediate_summaries)
    final_summary = await asyncio.to_thread(generate_with_llm, synthesis_prompt, model_name)

    return {
        "summary": final_summary,
        "chunkCount": len(docs),
        "intermediateCount": len(intermediate_summaries)
    }

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
    document_ids = data.get("documentIds") or []
    use_mcp_tools = data.get("use_agent_tools", False)  # Keep same param name for frontend compatibility
    lower = (prompt or "").lower()
    if any(k in lower for k in ["cite", "source", "references", "reference", "where did", "which paper", "according to"]):
        max_citations = 5
    else:
        max_citations = 0

    print(f"🎯 /generate → model={model}, mcp_tools={use_mcp_tools}, docs={len(document_ids)}")
    citations = []
    
    # === MCP TOOLS MODE ===
    if use_mcp_tools:
        try:
            ddg = DuckDuckGoSearchRun()
            tools_info = "You have access to:\n"
            tools_info += "1. web_search: Search the web using DuckDuckGo for current information\n"
            try:
                search_query = prompt
                ddg_results = ddg.run(search_query)
            except Exception as e:
                ddg_results = f"DuckDuckGo search failed: {e}"

            if document_ids:
                doc_content = get_document_content(document_ids)
                tools_info += "2. Uploaded research documents content (provided below)\n"
                enhanced_prompt = (
                    f"{tools_info}\n\nDuckDuckGo results:\n{ddg_results}\n\nDocument Content:\n{doc_content}\n\nUser Question: {prompt}\n\nPlease answer the question using the documents and web search if needed."
                )
            else:
                enhanced_prompt = (
                    f"{tools_info}\n\nDuckDuckGo results:\n{ddg_results}\n\nUser Question: {prompt}\n\nPlease use the web results to provide a comprehensive answer."
                )
            response_text = generate_with_llm(enhanced_prompt, "remote" if model.lower() == "remote" else model)
            return JSONResponse(content={
                "response": response_text,
                "citations": citations,
                "toolCalls": []
            })

        except Exception as e:
            print(f"❌ Web tools error: {e}")
            return JSONResponse(content={"error": f"Web tools error: {str(e)}"}, status_code=500)
    
    # === RAG MODE (Standard) ===
    if document_ids and max_citations > 0:
        context, citations = build_context_from_documents(document_ids, prompt, top_k=max_citations)
        if context:
            prompt = build_rag_prompt(context, prompt)

    # === GENERATION ===
    if model.lower() == "remote" and GOOGLE_API_KEY:
        response = generate_with_llm(prompt, model)
        return JSONResponse(content={"response": response, "citations": citations})
    else:
        response_text = generate_with_llm(prompt, model)
        return JSONResponse(content={"response": response_text, "citations": citations})


# === 🌍 NGROK ===
NGROK_AUTHTOKEN = "32eB7tLSQoICKJD4JSQuJ9lWea6_7U5ndjtQCVaWnPLEc4Mws"
FIXED_URL = "https://mari-unbequeathed-milkily.ngrok-free.app"
!ngrok config add-authtoken $NGROK_AUTHTOKEN

print("🌐 Starting ngrok tunnel...")
ngrok_proc = subprocess.Popen(["ngrok", "http", "--host-header=rewrite", "--log", "stdout", "--url", FIXED_URL, "8000"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
threading.Thread(target=stream_logs, args=(ngrok_proc, "ngrok"), daemon=True).start()
time.sleep(3)
print(f"✅ Public URL: {FIXED_URL}\n")

config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)
server = uvicorn.Server(config)
def run_uvicorn():
    asyncio.run(server.serve())

threading.Thread(target=run_uvicorn, daemon=True).start()
print("🚀 FastAPI running with live logs (Colab-safe)...\n")

while True:
    time.sleep(30)