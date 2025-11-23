import React, { useState } from "react";
import { Eye, X, BookOpen, Info } from "lucide-react";
import { Switch } from "./switch";
import { Button } from "./button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "./dialog";
import { Tooltip, TooltipContent, TooltipTrigger } from "./tooltip";
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
  onToggleEnabled?: (enabled: boolean) => void;
  onViewChunks?: (documentId: string, documentName: string) => void;
}

export const FileCard: React.FC<FileCardProps> = ({ file, onPreview, onRemove, onToggleEnabled, onViewChunks }) => {
  const [showMetadata, setShowMetadata] = useState(false);
  
  return (<>
  <div className="flex items-center gap-1 bg-background/50 border rounded-lg px-2 py-1 text-sm shrink-0 hover:bg-accent/50 transition-colors">
    <div className="flex items-center truncate max-w-[120px]">
      <StatusDot status={file.processingStatus || file.uploadStatus} />
      <a href={file.localUrl} target="_blank" rel="noopener noreferrer" className="truncate underline decoration-dotted">
        {file.name}
      </a>
    </div>
    <div className="flex items-center gap-1">
      {file.statusMessage && (
        <div className="text-xs text-muted-foreground mr-2">{file.statusMessage}</div>
      )}
      {onToggleEnabled && file.processingStatus === 'done' && (
        <div className="flex items-center gap-1 mr-1" title={file.enabled !== false ? "Enabled for context" : "Disabled"}>
          <Switch 
            checked={file.enabled !== false} 
            onCheckedChange={onToggleEnabled}
            className="scale-75"
          />
        </div>
      )}
      {file.processingStatus === 'done' && (
        <button
          onClick={() => setShowMetadata(true)}
          className="p-1 opacity-60 hover:opacity-100 rounded-sm hover:bg-background"
          title="View Metadata"
        >
          <Info className="h-3.5 w-3.5" />
        </button>
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
  
  <Dialog open={showMetadata} onOpenChange={setShowMetadata}>
    <DialogContent className="max-w-[95vw] sm:max-w-lg">
      <DialogHeader>
        <DialogTitle>Document Metadata</DialogTitle>
      </DialogHeader>
      <div className="space-y-3">
        <div className="grid grid-cols-[120px_1fr] gap-3 text-sm">
          <span className="text-muted-foreground font-medium">Name:</span>
          <span className="break-all">{file.name}</span>
        </div>
        <div className="grid grid-cols-[120px_1fr] gap-3 text-sm">
          <span className="text-muted-foreground font-medium">Type:</span>
          <span>{file.type || 'Unknown'}</span>
        </div>
        {file.documentId && (
          <div className="grid grid-cols-[120px_1fr] gap-3 text-sm">
            <span className="text-muted-foreground font-medium">Document ID:</span>
            <span className="font-mono text-xs break-all">{file.documentId}</span>
          </div>
        )}
        {file.processingStatus && (
          <div className="grid grid-cols-[120px_1fr] gap-3 text-sm">
            <span className="text-muted-foreground font-medium">Status:</span>
            <span className="capitalize">{file.processingStatus}</span>
          </div>
        )}
        {file.statusMessage && (
          <div className="grid grid-cols-[120px_1fr] gap-3 text-sm">
            <span className="text-muted-foreground font-medium">Message:</span>
            <span className="text-muted-foreground">{file.statusMessage}</span>
          </div>
        )}
        {file.chunkCount && (
          <div className="grid grid-cols-[120px_1fr] gap-3 text-sm">
            <span className="text-muted-foreground font-medium">Chunks:</span>
            <span>{file.chunkCount} chunks</span>
          </div>
        )}
      </div>
      {file.documentId && onViewChunks && (
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              onClick={() => {
                setShowMetadata(false);
                onViewChunks(file.documentId!, file.name);
              }}
              className="w-full mt-4"
            >
              <BookOpen className="h-4 w-4 mr-2" />
              View Chunks
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>View document chunks and content</p>
          </TooltipContent>
        </Tooltip>
      )}
    </DialogContent>
  </Dialog>
  </>
);};
