export interface FileWithUrl {
  name: string;
  url?: string;
  type?: string;
  size?: number;
  key?: string; 
  chatId?: string; 
  localUrl?: string;
  localFile?: File;
  documentId?: string;
  summary?: string;
  enabled?: boolean;
  uploadStatus?: 'idle' | 'uploading' | 'uploaded' | 'failed';
  downloadStatus?: 'idle' | 'downloading' | 'done' | 'failed';
  processingStatus?: 'idle' | 'processing' | 'done' | 'failed';
  statusMessage?: string;
  chunkCount?: number;
}
