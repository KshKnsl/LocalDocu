const NGROK_URL = process.env.NEXT_PUBLIC_BACKEND_URL;

export async function loadModel(modelName: string): Promise<{ status: string }> {
  const res = await fetch(`${NGROK_URL}/pull`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: modelName, stream: false }),
  });
  if (!res.ok) throw new Error(await res.text() || "Failed to load model");
  return res.json();
}

export async function sendExternalChatMessage({ prompt, model = "gemma3", stream = false, onStreamChunk }: { prompt: string; model?: string; stream?: boolean; onStreamChunk?: (chunk: string) => void }): Promise<{ response: string }> {
  await loadModel(model);
  const res = await fetch(`${NGROK_URL}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, prompt, stream }),
  });
  if (!res.ok) throw new Error(await res.text() || "Failed to get chat response");
  if (!stream) {
    const data = await res.json();
    return { response: data.response };
  }
  const reader = res.body?.getReader();
  const decoder = new TextDecoder();
  let result = "";
  if (!reader) throw new Error("No stream reader available");
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || ""; // keep incomplete line for next chunk
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const json = JSON.parse(line);
        if (json.response !== undefined) {
          result += json.response;
          if (onStreamChunk) onStreamChunk(json.response);
        }
      } catch (e) {
      }
    }
  }
  if (buffer.trim()) {
    try {
      const json = JSON.parse(buffer);
      if (json.response !== undefined) {
        result += json.response;
        if (onStreamChunk) onStreamChunk(json.response);
      }
    } catch (e) {}
  }
  return { response: result };
}

export type UploadResult = { url: string; filename: string; key: string };
export type ProcessingResult = { documentId: string; status: string; chunkCount: number; summary: string };
export type ChatResponse = { response: string };

export async function uploadDocument(file: File, opts?: { chatFolder?: string }): Promise<UploadResult> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("contentType", file.type);
  if (opts?.chatFolder) {
    formData.append("folder", opts.chatFolder);
  }
  const res = await fetch("/api/document/upload", { method: "POST", body: formData });
  if (!res.ok) throw new Error(await res.text() || "Failed to upload document");
  return res.json();
}

export async function processDocument(key?: string, file?: File): Promise<ProcessingResult> {
  let requestConfig: RequestInit;
  
  if (file) {
    const formData = new FormData();
    formData.append("file", file);
    requestConfig = {
      method: "POST",
      body: formData,
    };
  } else if (key) {
    requestConfig = {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key }),
    };
  } else {
    throw new Error("Either key or file must be provided");
  }

  const res = await fetch(`${NGROK_URL}/api/document/process`, requestConfig);
  if (!res.ok) throw new Error("Failed to process document");
  return res.json();
}

export async function summarizePaper(key: string): Promise<{ summary: string; paper_length: number }> {
  const res = await fetch(`${NGROK_URL}/api/summarize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key }),
  });
  if (!res.ok) throw new Error("Failed to summarize paper");
  return res.json();
}