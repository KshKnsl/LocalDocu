
export type UploadResult = { url: string; filename: string; key: string };
export type ProcessingResult = { documentId: string; status: string; chunkCount: number; summary: string };
export type ChatResponse = { response: string };
export type ChatMessage = { message: string; documentId?: string };

export async function uploadDocument(file: File): Promise<UploadResult> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("contentType", file.type);
  const res = await fetch("/api/document/upload", { method: "POST", body: formData });
  if (!res.ok) throw new Error(await res.text() || "Failed to upload document");
  return res.json();
}

export async function processDocument(url: string): Promise<ProcessingResult> {
  const res = await fetch("/api/document/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!res.ok) throw new Error("Failed to process document");
  return res.json();
}

export async function uploadAndProcessDocument(file: File): Promise<ProcessingResult> {
  const { url } = await uploadDocument(file);
  console.log(url);
  return processDocument(url);
}

export async function sendChatMessage({ message, documentId }: ChatMessage): Promise<ChatResponse> {
  const res = await fetch("https://6e43d38dae22.ngrok-free.app/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, documentId }),
  });
  return res.json();
}
