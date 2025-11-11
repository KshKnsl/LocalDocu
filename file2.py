!curl -fsSL https://ollama.com/install.sh | sh
!pip install fastapi uvicorn pyngrok requests boto3 python-multipart aiofiles langchain langchain-community chromadb sentence-transformers PyMuPDF langchain-huggingface langchain-chroma langchain-google-genai langchain-ollama
import os, signal, psutil, gc, time

os.system("pkill -f 'uvicorn' || true")
os.system("pkill -f 'ngrok' || true")
os.system("pkill -f 'ollama' || true")
gc.collect()
time.sleep(1)
print("✅ All background processes and threads terminated.")

import os, subprocess, threading, time, requests, tempfile, asyncio
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from pyngrok import ngrok
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama.llms import OllamaLLM
from uuid import uuid4
import uvicorn, sys
from google.colab import userdata
GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
from langchain_google_genai import ChatGoogleGenerativeAI
import base64
from pathlib import Path

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

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
OLLAMA_URL = "http://localhost:11434"
PERSIST_BASE = os.path.abspath("./chroma_store")
IMAGE_STORE = os.path.abspath("./image_store")
os.makedirs(IMAGE_STORE, exist_ok=True)
app = FastAPI(title="🦙 Ollama + LangChain Summarizer API")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_image_file(filename: str) -> bool:
    """Check if file is an image based on extension"""
    return Path(filename).suffix.lower() in IMAGE_EXTENSIONS

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

# ===============================
# Utility Functions
# ===============================
def generate_with_llm(prompt: str, model_name: str):
    """Unified LLM generation for both Gemini and Ollama"""
    if model_name.lower() == "remote":
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",api_key=GOOGLE_API_KEY)
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

# NOTE: Image summarization removed intentionally. Image queries are still supported via /generate.

# ===============================
# API Endpoints
# ===============================
@app.get("/")
def home():
    return {"message": "Ollama + LangChain API active"}

@app.post("/process")
async def process(file: UploadFile):
    if is_image_file(file.filename):
        doc_id = f"img_{uuid4().hex}"
        image_path = os.path.join(IMAGE_STORE, f"{doc_id}{Path(file.filename).suffix}")
        with open(image_path, "wb") as f:
            f.write(await file.read())
        print(f"🖼️ Image saved: {image_path}")
        return {
            "documentId": doc_id, 
            "status": "image_saved", 
            "isImage": True,
            "imagePath": image_path
        }
    
    chunks = load_and_split_pdf(await file.read())
    
    if len(chunks) == 0:
        return JSONResponse(
            status_code=400,
            content={
                "error": "No text content found in the document",
                "message": "This appears to be an image or has no extractable text. Please upload it as an image file (JPG, PNG, etc.) instead.",
                "chunkCount": 0
            }
        )
    
    doc_id = f"doc_{uuid4().hex}"
    create_persistent_vectorstore(chunks, os.path.join(PERSIST_BASE, doc_id))
    return {"documentId": doc_id, "status": "embeddings_created", "chunkCount": len(chunks), "isImage": False}

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

    # Helper: build final synthesis prompt
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

    from fastapi.responses import StreamingResponse
    import json
    
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
    
    # === PLAGIARISM CHECKER MODE (To be implemented) ===
    if use_mcp_tools:
        return JSONResponse(content={
            "response": "Plagiarism checker is not yet implemented.",
            "citations": [],
            "toolCalls": [],
            "status": "not_implemented"
        })
    
    image_ids = [doc_id for doc_id in document_ids if doc_id.startswith("img_")]
    text_ids = [doc_id for doc_id in document_ids if doc_id.startswith("doc_")]
    
    if image_ids:
        return await process_image_query(image_ids, text_ids, prompt, model)
    
    if text_ids:
        context, citations = build_context_from_documents(text_ids, prompt, top_k=max_citations)
        prompt = build_rag_prompt(context, prompt)

    response_text = generate_with_llm(prompt, model)
    return JSONResponse(content={"response": response_text, "citations": citations})

async def process_image_query(image_ids: list, text_ids: list, prompt: str, model: str):
    """Process queries with image context using LLaVA"""
    vision_model = "llava" if model.lower() != "remote" else "llava"
    
    responses = []
    for img_id in image_ids:
        image_files = [f for f in os.listdir(IMAGE_STORE) if f.startswith(img_id)]
        if not image_files:
            responses.append(f"Image {img_id} not found.")
            continue
        
        image_path = os.path.join(IMAGE_STORE, image_files[0])
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": vision_model,
                    "prompt": prompt,
                    "images": [image_data],
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                responses.append(result.get("response", "No response from vision model"))
            else:
                responses.append(f"Error: Vision model returned status {response.status_code}")
        except Exception as e:
            responses.append(f"Error processing image: {str(e)}")
    
    additional_context = ""
    citations = []
    if text_ids:
        context, citations = build_context_from_documents(text_ids, prompt, top_k=3)
        additional_context = f"\n\nAdditional context from documents:\n{context}"
    
    final_response = "\n\n".join(responses)
    if additional_context:
        final_response += additional_context
    
    return JSONResponse(content={
        "response": final_response,
        "citations": citations,
        "usedVisionModel": True,
        "visionModel": vision_model
    })


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