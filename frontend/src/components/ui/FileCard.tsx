import React from "react";
import { Eye, X } from "lucide-react";
import type { FileWithUrl } from "./FileWithUrl";

interface FileCardProps {
  file: FileWithUrl;
  onPreview?: (file: FileWithUrl) => void;
  onRemove?: () => void;
}

export const FileCard: React.FC<FileCardProps> = ({ file, onPreview, onRemove }) => (
  <div className="flex items-center gap-1 bg-background/50 border rounded-lg px-2 py-1 text-sm shrink-0 hover:bg-accent/50 transition-colors">
    <a href={file.url} target="_blank" rel="noopener noreferrer" className="truncate max-w-[100px] underline decoration-dotted">
      {file.name}
    </a>
    <div className="flex items-center">
      {onPreview && (
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
