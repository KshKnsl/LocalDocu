const getBackendUrl = () => typeof window !== "undefined" && localStorage.getItem("backendUrl") || "https://minor-project-6v6z.vercel.app/api";
const getProgressServiceUrl = () => "https://minor-project-progress.vercel.app";
export const isUsingCustomBackend = () => typeof window !== "undefined" && localStorage.getItem("backendMode") === "custom" && !!localStorage.getItem("backendUrl");

export const loadModel = async (modelName: string) => {
  const res = await fetch(`${getBackendUrl()}/pull`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: modelName, stream: false }) });
  if (!res.ok) throw new Error(await res.text() || "Failed to load model");
  return res.json();
};

export async function sendExternalChatMessage({ prompt, model = "mistral", stream = false, documentIds, maxCitations, specificChunks, onStreamChunk, onStatusChange }: { prompt: string; model?: string; stream?: boolean; documentIds?: string[]; maxCitations?: number | null; specificChunks?: Record<string, number[]>; onStreamChunk?: (chunk: string) => void; onStatusChange?: (status: string) => void }): Promise<ChatResponse> {
  if (onStatusChange) onStatusChange("Pulling model...");
  await loadModel(model);
  if (onStatusChange) onStatusChange("Generating...");
  const body: any = { model, prompt, stream, documentIds };
  if (typeof maxCitations !== "undefined") body.maxCitations = maxCitations;
  if (specificChunks) body.specificChunks = specificChunks;

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

export const getProgress = async (documentId?: string) => {
  try {
    const url = documentId ? `${getProgressServiceUrl()}/progress?documentId=${encodeURIComponent(documentId)}` : `${getProgressServiceUrl()}/progress`;
    const res = await fetch(url);
    return res.ok ? (await res.json()).progress : null;
  } catch { return null; }
};

export const clearProgress = async (documentId: string) => {
  try { await fetch(`${getProgressServiceUrl()}/progress/${encodeURIComponent(documentId)}`, { method: "DELETE" }); } catch {}
};

export const uploadDocument = async (file: File, opts?: { chatFolder?: string }): Promise<UploadResult> => {
  return { url: "", filename: file.name, key: `local-${Date.now()}-${file.name}` };
};

export async function processDocument(key?: string, file?: File): Promise<ProcessingResult> {
  if (!file) {
    // No file provided, skip processing
    return { documentId: key || "", status: "skipped", chunkCount: 0 };
  }

  const formData = new FormData();
  formData.append("file", file);

  let pollTimer: number | null = null;
  if (typeof window !== "undefined" && !isUsingCustomBackend()) {
    pollTimer = window.setInterval(async () => {
      try {
        const progress = await getProgress();
        // If progress service returns null, the work has already completed
        // and the progress record was removed — stop polling.
        if (progress === null) {
          if (pollTimer !== null) {
            clearInterval(pollTimer);
            pollTimer = null;
          }
          return;
        }
      } catch (e) {
        // ignore polling errors
      }
    }, 1000);
  }

  try {
    const res = await fetch(`${getBackendUrl()}/process`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(await res.text() || "Failed to process document");
    return res.json();
  } finally {
    if (pollTimer !== null) {
      clearInterval(pollTimer);
    }
  }
}

export type DocumentChunk = {
  id: number;
  content: string;
  metadata: Record<string, any>;
  images?: Array<{
    id: string;
    url: string;
    summary: string;
    page: number;
  }>;
};

export const getDocumentChunks = async (documentId: string): Promise<DocumentChunk[]> => {
  const res = await fetch(`${getBackendUrl()}/get_chunks`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ documentId }) });
  if (!res.ok) throw new Error(await res.text() || "Failed to get chunks");
  return (await res.json()).chunks;
};

