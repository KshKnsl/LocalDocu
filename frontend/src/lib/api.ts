export interface UploadResult {
  url: string;
  filename: string;
  key: string;
}

export interface ProcessingResult {
  documentId: string;
  status: string;
  chunkCount: number;
  summary: string;
}

export interface ChatResponse {
  response: string;
}

export interface ChatMessage {
  message: string;
  documentId?: string;
}

export async function uploadDocument(file: File): Promise<UploadResult> {
  try {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("contentType", file.type);

    const response = await fetch("/api/document/upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(error || "Failed to upload document");
    }

    return response.json();
  } catch (error) {
    console.error("Upload error:", error);
    throw error instanceof Error ? error : new Error("Failed to upload document");
  }
}

export async function processDocument(url: string): Promise<ProcessingResult> {
  const response = await fetch("/api/document/process", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ url }),
  });

  if (!response.ok) {
    throw new Error("Failed to process document");
  }

  return response.json();
}

export async function uploadAndProcessDocument(file: File): Promise<ProcessingResult> {
  try {
    const uploadResult = await uploadDocument(file);
    console.log('File uploaded successfully:', {
      url: uploadResult.url,
      filename: uploadResult.filename,
      key: uploadResult.key
    });
    
    return processDocument(uploadResult.url);
  } catch (error) {
    console.error('Error in uploadAndProcessDocument:', error);
    throw error;
  }
}

export async function sendChatMessage({
  message,
  documentId,
}: ChatMessage): Promise<ChatResponse> {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message, documentId }),
  });
  const data = await response.json();
  return data;
}
