import { FileWithUrl } from "@/components/ui/FileWithUrl";
import { uploadDocument, processDocument, getProgress, clearProgress, isUsingCustomBackend, ProgressData, ProcessingResult } from "@/lib/api";
import { ProcessingFile } from "./types";
import { toast } from "sonner";

export async function handleFileUpload(
  files: File[],
  chatId: string,
  chatFolder: string,
  setFiles: React.Dispatch<React.SetStateAction<FileWithUrl[]>>,
  setProcessingFiles: React.Dispatch<React.SetStateAction<ProcessingFile[]>>,
  setIsProcessing: React.Dispatch<React.SetStateAction<boolean>>,
  setShowProcessingDialog: React.Dispatch<React.SetStateAction<boolean>>
): Promise<void> {
  const fileArray = Array.from(files);
  const tempIds = fileArray.map(() => `temp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  
  setIsProcessing(true);
  setShowProcessingDialog(true);
  setProcessingFiles(fileArray.map(file => ({
    name: file.name,
    status: 'uploading',
    progress: 0,
  })));
  
  const placeholders: FileWithUrl[] = fileArray.map((file, idx) => ({
    name: file.name,
    type: file.type,
    size: file.size,
    chatId,
    enabled: true, 
    uploadStatus: 'uploading',
    processingStatus: 'idle',
    downloadStatus: 'idle',
    statusMessage: 'Uploading',
    key: tempIds[idx],
  }));
  setFiles((prev) => [...prev, ...placeholders]);

  const uploaded: FileWithUrl[] = await Promise.all(
    fileArray.map(async (file, idx) => {
      const tempId = tempIds[idx];
      try {
        const { url, key, filename } = await uploadDocument(file, { chatFolder });
        
        setProcessingFiles(prev => prev.map(pf => 
          pf.name === file.name && pf.status === 'uploading'
            ? { ...pf, status: 'processing' as const, progress: 10 }
            : pf
        ));
        
        setFiles((prev) => prev.map(f => (f.key === tempId && f.chatId === chatId ? {
          ...f,
          url,
          key,
          name: filename || f.name,
          uploadStatus: 'uploaded',
          statusMessage: isUsingCustomBackend() ? 'Ready' : 'Uploaded',
          localFile: isUsingCustomBackend() ? file : undefined,
        } : f)));
        return {
          name: filename || file.name,
          url,
          key,
          type: file.type,
          size: file.size,
          chatId,
          enabled: true,
          uploadStatus: 'uploaded',
          statusMessage: isUsingCustomBackend() ? 'Ready' : 'Uploaded',
          localFile: isUsingCustomBackend() ? file : undefined,
        } as FileWithUrl;
      } catch (err) {
        setFiles((prev) => prev.map(f => (f.key === tempId && f.chatId === chatId ? { ...f, uploadStatus: 'failed', statusMessage: 'Upload failed' } : f)));
        return {
          name: file.name,
          type: file.type,
          size: file.size,
          chatId,
          uploadStatus: 'failed',
          statusMessage: 'Upload failed',
        } as FileWithUrl;
      }
    })
  );

  const documentIdByKey = new Map<string, string>();
  for (const uploadedFile of uploaded) {
    if (!uploadedFile.key) {
      setFiles((prev) => prev.map(f => (f.name === uploadedFile.name && f.chatId === chatId ? { ...f, processingStatus: 'failed', statusMessage: 'Missing key' } : f)));
      continue;
    }
    setFiles((prev) => prev.map(f => (f.key === uploadedFile.key && f.chatId === chatId ? { ...f, processingStatus: 'processing', statusMessage: 'Processing' } : f)));
    
    setProcessingFiles(prev => prev.map(pf => 
      pf.name === uploadedFile.name && pf.status === 'processing'
        ? { ...pf, progress: 30 }
        : pf
    ));
    
    try {
      let result: ProcessingResult | undefined;
      if (isUsingCustomBackend() && uploadedFile.localFile) {
        result = await processDocument(undefined, uploadedFile.localFile);
      } else if (uploadedFile.url) {
        try {
          const resp = await fetch(uploadedFile.url);
          const blob = await resp.blob();
          const fileObj = new File([blob], uploadedFile.name, { type: uploadedFile.type || "application/pdf" });
          result = await processDocument(undefined, fileObj);
        } catch (err) {
          result = await processDocument(uploadedFile.key);
        }
      } else {
        result = await processDocument(uploadedFile.key);
      }
      if (result?.documentId) {
        documentIdByKey.set(uploadedFile.key, result.documentId);
        
        let attempts = 0;
        const maxAttempts = 300; 
        let lastProgress = 30;
        
        while (attempts < maxAttempts) {
          const progressData = await getProgress(result.documentId) as ProgressData | null;
          
          if (progressData) {
            const progress = progressData.progress || lastProgress;
            lastProgress = progress;
            
            setProcessingFiles(prev => prev.map(pf => 
              pf.name === uploadedFile.name
                ? { 
                    ...pf, 
                    progress,
                    chunks: progressData.totalChunks,
                    currentChunk: progressData.currentChunk,
                    status: progressData.status === 'complete' ? 'done' as const :
                           progressData.status === 'failed' ? 'failed' as const : 
                           'processing' as const
                  }
                : pf
            ));
            
            if (progressData.status === 'complete' || progress >= 100) {
              setProcessingFiles(prev => prev.map(pf => 
                pf.name === uploadedFile.name
                  ? { ...pf, status: 'done' as const, progress: 100 }
                  : pf
              ));
              await clearProgress(result.documentId);
              break;
            }
            
            if (progressData.status === 'failed') {
              throw new Error('Processing failed');
            }
          }
          
          await new Promise(resolve => setTimeout(resolve, 100));
          attempts++;
        }
        
        setFiles((prev) => prev.map(f => (f.key === uploadedFile.key && f.chatId === chatId ? { ...f, documentId: result.documentId, processingStatus: 'done', statusMessage: 'Processed' } : f)));
        toast.success(`Document ${uploadedFile.name} processed for RAG`, { duration: 2000 });
      } else {
        setProcessingFiles(prev => prev.map(pf => 
          pf.name === uploadedFile.name
            ? { ...pf, status: 'failed' as const }
            : pf
        ));
        setFiles((prev) => prev.map(f => (f.key === uploadedFile.key && f.chatId === chatId ? { ...f, processingStatus: 'failed', statusMessage: 'Processing failed' } : f)));
        toast.error(`Failed to process ${uploadedFile.name}`);
      }
    } catch (error) {
      setProcessingFiles(prev => prev.map(pf => 
        pf.name === uploadedFile.name
          ? { ...pf, status: 'failed' as const }
          : pf
      ));
      setFiles((prev) => prev.map(f => (f.key === uploadedFile.key && f.chatId === chatId ? { ...f, processingStatus: 'failed', statusMessage: 'Processing error' } : f)));
      toast.error(`Failed to process ${uploadedFile.name}: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
}
