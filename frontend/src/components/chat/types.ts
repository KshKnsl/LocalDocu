import { FileWithUrl } from "@/components/ui/FileWithUrl";
import { MessageObject } from "@/lib/chatStorage";

export interface ProcessingFile {
  name: string;
  status: 'uploading' | 'processing' | 'done' | 'failed';
  progress?: number;
  chunks?: number;
  currentChunk?: number;
}

export interface ChatState {
  currentChatId?: string;
  input: string;
  files: FileWithUrl[];
  model: string;
  processingFiles: ProcessingFile[];
  showProcessingDialog: boolean;
  isProcessing: boolean;
  selectedChunks: Record<string, number[]>;
}

export interface ChatHandlers {
  onFileSelect: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onRemoveFile: (index: number) => void;
  onSubmit: () => void;
  setInput: (value: string) => void;
  setModel: (model: string) => void;
}
