import type { FileWithUrl } from "@/components/ui/FileWithUrl";

export type FileType = "image" | "pdf" | "video" | "text" | "other";

export const getFileType = (input: FileWithUrl | string): FileType => {
  const name = (typeof input === "string" ? input : input.name).toLowerCase();
  if (/\.(jpg|jpeg|png|gif|webp|svg|bmp)$/.test(name)) return "image";
  if (name.endsWith(".pdf")) return "pdf";
  if (/\.(mp4|webm|ogg|mov|avi)$/.test(name)) return "video";
  if (/\.(txt|md|json|csv|log|xml|html|css|js|ts|tsx|jsx|py|java|c|cpp|h)$/.test(name)) return "text";
  return "other";
};

export const downloadFile = (url: string, filename?: string) => {
  const a = Object.assign(document.createElement("a"), { href: url, download: filename || url.split("/").pop() || "download" });
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
};

export const formatFileSize = (bytes: number) => {
  if (!bytes) return "0 Bytes";
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${Math.round(bytes / Math.pow(1024, i) * 100) / 100} ${["Bytes", "KB", "MB", "GB"][i]}`;
};
