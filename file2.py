# === 1️⃣ INSTALL DEPENDENCIES ===
!curl -fsSL https://ollama.com/install.sh | sh
!pip install fastapi uvicorn pyngrok requests boto3 python-multipart aiofiles langchain langchain-community chromadb sentence-transformers PyMuPDF

# === 🔥 HARD CLEANUP CELL (use before running the API) ===
import os, signal, psutil, gc, time

print("🧹 Killing all existing Ollama / FastAPI / ngrok processes...")

for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        cmdline = " ".join(proc.info['cmdline'] or [])
        if any(word in cmdline for word in ["ollama", "uvicorn", "ngrok", "fastapi"]):
            print(f"🛑 Killing PID {proc.pid}: {cmdline}")
            proc.kill()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

# Also clear leftover threads and sockets
os.system("pkill -f 'uvicorn' || true")
os.system("pkill -f 'ngrok' || true")
os.system("pkill -f 'ollama' || true")

# Garbage collect and wait a moment
gc.collect()
time.sleep(2)
print("✅ All background processes and threads terminated.")

import os, subprocess, threading, time, requests, tempfile, asyncio
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import JSONResponse
from pyngrok import ngrok
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.llms import Ollama
import uvicorn, sys

print("🧹 Cleaned old processes.\n")

# === 🔁 LOG STREAMER ===
def stream_logs(proc, name):
    for line in iter(proc.stdout.readline, b''):
        sys.stdout.write(f"[{name}] {line.decode()}")
        sys.stdout.flush()
    for line in iter(proc.stderr.readline, b''):
        sys.stdout.write(f"[{name}-ERR] {line.decode()}")
        sys.stdout.flush()

# === 🦙 START OLLAMA LOCALLY ===
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
OLLAMA_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434"
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

def create_vectorstore(chunks):
    print("🧠 Creating vectorstore...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(chunks, embeddings)

def summarize_text(chunks):
    text = "\n".join(c.page_content for c in chunks)
    prompt = f"Summarize in about 200 words:\n{text[:8000]}"
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={"model": OLLAMA_MODEL, "prompt": prompt})
    print(f"✅ Summary generated ({r.status_code}).")
    return r.json().get("response", "No summary.")

def query_doc(vs, question):
    retriever = vs.as_retriever(search_kwargs={"k":5})
    docs = retriever.get_relevant_documents(question)
    ctx = "\n".join(d.page_content for d in docs)
    llm = Ollama(model=OLLAMA_MODEL)
    return llm.invoke(f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:")

@app.get("/")
def home():
    print("📡 Health check called.")
    return {"message": "✅ Ollama + LangChain API active on Colab"}

@app.post("/summarize")
async def summarize(file: UploadFile):
    data = await file.read()
    chunks = load_and_split_pdf(data)
    return {"summary": summarize_text(chunks)}

@app.post("/query")
async def query(file: UploadFile, question: str = Form(...)):
    data = await file.read()
    chunks = load_and_split_pdf(data)
    vs = create_vectorstore(chunks)
    return {"answer": query_doc(vs, question)}

@app.post("/pull")
async def pull_model(request: Request):
    try:
        data = await request.json()
        model = "gemma3"
        print(f"📥 Pulling model '{model}' from Ollama...")
        resp = requests.post(f"{OLLAMA_URL}/api/pull", json={"name": model, "stream": False})
        print(f"✅ Pull complete ({resp.status_code}).")
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except Exception as e:
        print(f"❌ Error pulling model: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.api_route("/api/{path:path}", methods=["GET", "POST"])
async def proxy_ollama(path: str, request: Request):
    try:
        body = await request.body()
        r = requests.request(request.method, f"{OLLAMA_URL}/api/{path}", data=body)
        return JSONResponse(content=r.json() if r.content else {}, status_code=r.status_code)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# === 🌍 NGROK ===
NGROK_AUTHTOKEN = "32eB7tLSQoICKJD4JSQuJ9lWea6_7U5ndjtQCVaWnPLEc4Mws"
FIXED_URL = "https://mari-unbequeathed-milkily.ngrok-free.app"
!ngrok config add-authtoken $NGROK_AUTHTOKEN

print("🌐 Starting ngrok tunnel...")
ngrok_proc = subprocess.Popen(["ngrok", "http", "--log", "stdout", "--url", FIXED_URL, "8000"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
threading.Thread(target=stream_logs, args=(ngrok_proc, "ngrok"), daemon=True).start()
time.sleep(3)
print(f"✅ Public URL: {FIXED_URL}\n")

# === 🚀 RUN FASTAPI WITH LIVE LOGS ===
config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info", access_log=True)
server = uvicorn.Server(config)

def run_uvicorn():
    asyncio.run(server.serve())

threading.Thread(target=run_uvicorn, daemon=True).start()
print("🚀 FastAPI running with live logs (Colab-safe)...\n")

# Keep the cell alive to show continuous logs
while True:
    time.sleep(30)
