import { saveChatFileToLocal } from "./localFiles";

const getBackendUrl = () => typeof window !== "undefined" && localStorage.getItem("backendUrl") || "https://localdocu-proxy.vercel.app/api";
const getProgressServiceUrl = () => "https://localdocu-progress.vercel.app";
export const isUsingCustomBackend = () => typeof window !== "undefined" && localStorage.getItem("backendMode") === "custom" && !!localStorage.getItem("backendUrl");

export const loadModel = async (modelName: string) => {
  const res = await fetch(`${getBackendUrl()}/pull`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: modelName, stream: false }) });
  if (!res.ok) throw new Error(await res.text() || "Failed to load model");
  return res.json();
};

export async function sendExternalChatMessage({ prompt, model = "mistral", documentIds, maxCitations, specificChunks }: { prompt: string; model?: string; documentIds?: string[]; maxCitations?: number | null; specificChunks?: Record<string, number[]>; }): Promise<ChatResponse> {
  await loadModel(model);
  const body: any = { model, prompt, documentIds };
  if (typeof maxCitations !== "undefined") body.maxCitations = maxCitations;
  if (specificChunks) body.specificChunks = specificChunks;

  const res = await fetch(`${getBackendUrl()}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text() || "Failed to get chat response");
  const data = await res.json();
  return { response: data.response, citations: data.citations };
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
  const url = documentId ? `${getProgressServiceUrl()}/progress?documentId=${encodeURIComponent(documentId)}` : `${getProgressServiceUrl()}/progress`;
  const res = await fetch(url);
  return res.ok ? (await res.json()).progress : null;
};

export const clearProgress = async (documentId: string) => {
  await fetch(`${getProgressServiceUrl()}/progress/${encodeURIComponent(documentId)}`, { method: "DELETE" });
};

export const uploadDocument = async (file: File, opts?: { chatFolder?: string }): Promise<UploadResult> => {
  const chatId = opts?.chatFolder?.replace('chats/', '') || 'temp';
  const localUrl = await saveChatFileToLocal(chatId, file.name, file, file.type);
  
  return { 
    url: localUrl, 
    filename: file.name, 
    key: `local-${Date.now()}-${file.name}` 
  };
};

export async function processDocument(key?: string, file?: File): Promise<ProcessingResult> {
  if (!file) {
    return { documentId: key || "", status: "skipped", chunkCount: 0 };
  }

  const formData = new FormData();
  formData.append("file", file);

  let pollTimer: number | null = null;
  if (typeof window !== "undefined" && !isUsingCustomBackend()) {
    pollTimer = window.setInterval(async () => {
      try {
        const progress = await getProgress();
        if (progress === null) {
          if (pollTimer !== null) {
            clearInterval(pollTimer);
            pollTimer = null;
          }
          return;
        }
      } catch (e) {
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

