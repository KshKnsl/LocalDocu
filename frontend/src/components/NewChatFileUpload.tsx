"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { ProcessingResult, uploadAndProcessDocument } from "@/lib/api";
import { Upload } from "lucide-react";
import { FileUploadArea } from "./ui/file-upload-area";
import { FileList } from "./ui/file-list";

interface NewChatFileUploadProps {
  onProcessingComplete?: (result: ProcessingResult) => void;
}

export function NewChatFileUpload({ onProcessingComplete }: NewChatFileUploadProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) setFiles(Array.from(e.target.files));
  };

  const handleProcess = async () => {
    if (!files.length) return;
    setLoading(true);
    const results = await Promise.all(files.map(uploadAndProcessDocument));
    results.forEach(result => onProcessingComplete?.(result));
    setFiles([]);
    setLoading(false);
  };

  return (
    <div className="w-full max-w-xl mx-auto space-y-4">
      <FileUploadArea
        files={files}
        onChange={handleFileChange}
        loading={loading}
        accept=".pdf,.doc,.docx,.txt"
        id="new-chat-file-upload"
      />
      {files.length > 0 && (
        <div className="space-y-3">
          <FileList files={files} onRemove={i => setFiles(f => f.filter((_, idx) => idx !== i))} />
          <Button onClick={handleProcess} disabled={loading} className="w-full">
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin mr-2" />
                Processing Files...
              </>
            ) : (
              <>
                <Upload className="w-4 h-4 mr-2" />
                Process {files.length} {files.length === 1 ? "File" : "Files"}
              </>
            )}
          </Button>
        </div>
      )}
    </div>
  );
}
