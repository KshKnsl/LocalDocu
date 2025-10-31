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

export async function sendExternalChatMessage({ prompt, model = "mistral", stream = false, documentIds, useAgentTools = false, maxCitations, onStreamChunk, onStatusChange }: { prompt: string; model?: string; stream?: boolean; documentIds?: string[]; useAgentTools?: boolean; maxCitations?: number | null; onStreamChunk?: (chunk: string) => void; onStatusChange?: (status: string) => void }): Promise<ChatResponse> {
  if (onStatusChange) onStatusChange("Pulling model...");
  await loadModel(model);
  if (onStatusChange) onStatusChange(useAgentTools ? "Running agent with tools..." : "Generating...");
  const body: any = { model, prompt, stream, documentIds, use_agent_tools: useAgentTools };
  if (typeof maxCitations !== "undefined") body.maxCitations = maxCitations;

  const res = await fetch(`${NGROK_URL}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text() || "Failed to get chat response");
  if (!stream) {
    const data = await res.json();
    return { response: data.response, citations: data.citations };
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
  return { response: result, citations: [] };
}

export type UploadResult = { url: string; filename: string; key: string };
export type ProcessingResult = { documentId: string; status: string; chunkCount: number };
export type Citation = { documentId: string; page: string | number; snippet: string; fullText: string; source: string; rank: number };
export type ChatResponse = { response: string; citations?: Citation[] };

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
  const formData = new FormData();
  if (file) {
    formData.append("file", file);
  } else if (key) {
    const downloadRes = await fetch(`/api/document/download?key=${encodeURIComponent(key)}`);
    if (!downloadRes.ok) throw new Error(await downloadRes.text() || "Failed to download document for processing");
    const blob = await downloadRes.blob();
    const cd = downloadRes.headers.get("Content-Disposition") || "";
    const match = cd.match(/filename="?([^";]+)"?/i);
    const filename = match?.[1] || `document-${Date.now()}`;
    formData.append("file", new File([blob], filename, { type: blob.type || "application/octet-stream" }));
  } else {
    throw new Error("Either key or file must be provided");
  }

  const res = await fetch(`${NGROK_URL}/process`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(await res.text() || "Failed to process document");
  return res.json();
}

export async function summarizeByDocumentId(documentId: string): Promise<{ summary: string }> {
  const res = await fetch(`${NGROK_URL}/summarize_by_id`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ documentId }),
  });
  if (!res.ok) throw new Error(await res.text() || "Failed to summarize document");
  return res.json();
}

