const getBackendUrl = () => {
  if (typeof window !== "undefined") {
    const customUrl = localStorage.getItem("backendUrl")
    if (customUrl) return customUrl
  }
  return "https://minor-project-6v6z.vercel.app/api"
}

const getProgressServiceUrl = () => {
  return "https://minor-project-progress.vercel.app/api"
}

export const isUsingCustomBackend = () => {
  if (typeof window !== "undefined") {
    const customUrl = localStorage.getItem("backendUrl")
    const backendMode = localStorage.getItem("backendMode")
    return backendMode === "custom" && !!customUrl
  }
  return false
}

export async function loadModel(modelName: string): Promise<{ status: string }> {
  const res = await fetch(`${getBackendUrl()}/pull`, {
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

  const res = await fetch(`${getBackendUrl()}/generate`, {
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
export type Citation = { documentId: string; page: string | number; snippet: string; fullText: string; source: string; rank: number; score?: number };
export type ChatResponse = { response: string; citations?: Citation[] };
export type ProgressData = {
  documentId: string;
  status: string;
  progress: number;
  message?: string;
  currentChunk?: number;
  totalChunks?: number;
};

export async function getProgress(documentId?: string): Promise<Record<string, ProgressData> | ProgressData | null> {
  try {
    const url = documentId 
      ? `${getProgressServiceUrl()}/progress?documentId=${encodeURIComponent(documentId)}`
      : `${getProgressServiceUrl()}/progress`;
    
    const res = await fetch(url);
    if (!res.ok) return null;
    
    const data = await res.json();
    return documentId ? data.progress : data.progress;
  } catch (e) {
    console.error("Failed to get progress:", e);
    return null;
  }
}

export async function clearProgress(documentId: string): Promise<void> {
  try {
    await fetch(`${getProgressServiceUrl()}/progress/${encodeURIComponent(documentId)}`, {
      method: "DELETE"
    });
  } catch (e) {
    console.error("Failed to clear progress:", e);
  }
}

export async function uploadDocument(file: File, opts?: { chatFolder?: string }): Promise<UploadResult> {
  if (isUsingCustomBackend()) {
    return {
      url: "",
      filename: file.name,
      key: `local-${Date.now()}-${file.name}`
    };
  }
  
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

  const res = await fetch(`${getBackendUrl()}/process`, { method: "POST", body: formData });
  if (!res.ok) throw new Error(await res.text() || "Failed to process document");
  return res.json();
}

export async function summarizeByDocumentId(
  documentId: string, 
  onChunkSummary?: (chunkIndex: number, total: number, summary: string) => void,
  onStatus?: (status: string) => void,
  model?: string
): Promise<{ summary: string; chunkCount?: number }> {
  const res = await fetch(`${getBackendUrl()}/summarize_by_id`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ documentId, model_name: model || "mistral" }),
  });
  if (!res.ok) {
    const errorText = await res.text();
    console.error("Summarize error:", errorText);
    throw new Error(errorText || "Failed to summarize document");
  }
  
  const reader = res.body?.getReader();
  if (!reader) {
    return res.json();
  }
  
  const decoder = new TextDecoder();
  let buffer = "";
  let finalSummary = "";
  let chunkCount = 0;
  
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || "";
    
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const data = JSON.parse(line);
        
        if (data.type === "error") {
          console.error("Backend error:", data.message);
          throw new Error(data.message + (data.suggestion ? `\n${data.suggestion}` : ""));
        } else if (data.type === "chunk" && onChunkSummary) {
          onChunkSummary(data.index, data.total, data.summary);
        } else if (data.type === "status" && onStatus) {
          onStatus(data.message);
        } else if (data.type === "final") {
          finalSummary = data.summary;
          chunkCount = data.chunkCount;
        }
      } catch (e) {
        if (e instanceof Error && e.message.includes("Backend error")) {
          throw e; // Re-throw backend errors
        }
        console.error("Failed to parse stream line:", e, "Line:", line);
      }
    }
  }
  
  // Process any remaining buffer
  if (buffer.trim()) {
    try {
      const data = JSON.parse(buffer);
      if (data.type === "error") {
        throw new Error(data.message + (data.suggestion ? `\n${data.suggestion}` : ""));
      } else if (data.type === "final") {
        finalSummary = data.summary;
        chunkCount = data.chunkCount;
      }
    } catch (e) {
      if (e instanceof Error && e.message.includes("Backend error")) {
        throw e;
      }
      console.error("Failed to parse final buffer:", e);
    }
  }
  
  if (!finalSummary) {
    throw new Error("No summary received from backend");
  }
  
  return { summary: finalSummary, chunkCount };
}

