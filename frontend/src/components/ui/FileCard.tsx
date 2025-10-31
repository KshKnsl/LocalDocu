import React from "react";
import { Eye, X } from "lucide-react";
import type { FileWithUrl } from "./FileWithUrl";

function StatusDot({ status }: { status?: string }) {
  const color = status === 'uploading' || status === 'processing' || status === 'downloading'
    ? 'bg-yellow-400'
    : status === 'failed'
      ? 'bg-destructive'
      : status === 'done' || status === 'uploaded'
        ? 'bg-green-400'
        : 'bg-muted-foreground/40';
  return <span className={`inline-block w-2 h-2 rounded-full ${color} mr-2`} />;
}

interface FileCardProps {
  file: FileWithUrl;
  onPreview?: (file: FileWithUrl) => void;
  onRemove?: () => void;
}

export const FileCard: React.FC<FileCardProps> = ({ file, onPreview, onRemove }) => (
  <div className="flex items-center gap-1 bg-background/50 border rounded-lg px-2 py-1 text-sm shrink-0 hover:bg-accent/50 transition-colors">
    <div className="flex items-center truncate max-w-[120px]">
      <StatusDot status={file.processingStatus || file.uploadStatus || file.downloadStatus} />
      <a href={file.localUrl} target="_blank" rel="noopener noreferrer" className="truncate underline decoration-dotted">
        {file.name}
      </a>
    </div>
    <div className="flex items-center">
      {file.statusMessage && (
        <div className="text-xs text-muted-foreground mr-2">{file.statusMessage}</div>
      )}
      {onPreview && file.processingStatus === 'done' && (
        <button
          onClick={() => onPreview(file)}
          className="p-1 opacity-60 hover:opacity-100 rounded-sm hover:bg-background"
          title="Preview"
        >
          <Eye className="h-3.5 w-3.5" />
        </button>
      )}
      {onRemove && (
        <button
          onClick={onRemove}
          className="p-1 text-destructive/60 hover:text-destructive rounded-sm hover:bg-background"
          title="Remove"
        >
          <X className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  </div>
);
