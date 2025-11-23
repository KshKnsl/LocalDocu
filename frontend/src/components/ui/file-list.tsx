'use client';

import { cn } from '@/lib/utils';
import { Eye, X, BookOpen, FileText, CheckCircle2, Loader2, XCircle, Clock } from 'lucide-react';
import { Switch } from './switch';
import { Button } from './button';
import { Badge } from './badge';
import { ScrollArea } from './scroll-area';
import type { FileWithUrl } from "./FileWithUrl";

interface FileListProps {
  files: FileWithUrl[];
  onRemove: (index: number) => void;
  previewFile?: (file: FileWithUrl) => void;
  onToggleEnabled?: (index: number, enabled: boolean) => void;
  onViewChunks?: (documentId: string, documentName: string) => void;
}

function StatusBadge({ status }: { status?: string }) {
  if (status === 'uploading' || status === 'processing' || status === 'downloading') {
    return (
      <Badge variant="secondary" className="gap-1.5">
        <Loader2 className="h-3 w-3 animate-spin" />
        <span className="capitalize">{status}</span>
      </Badge>
    );
  }
  if (status === 'failed') {
    return (
      <Badge variant="destructive" className="gap-1.5">
        <XCircle className="h-3 w-3" />
        Failed
      </Badge>
    );
  }
  if (status === 'done' || status === 'uploaded') {
    return (
      <Badge variant="default" className="gap-1.5 bg-green-600 hover:bg-green-700">
        <CheckCircle2 className="h-3 w-3" />
        Done
      </Badge>
    );
  }
  return (
    <Badge variant="outline" className="gap-1.5">
      <Clock className="h-3 w-3" />
      Idle
    </Badge>
  );
}

export function FileList({ files, onRemove, previewFile, onToggleEnabled, onViewChunks }: FileListProps) {
  console.log('FileList props:', { onRemove: !!onRemove, previewFile: !!previewFile, onToggleEnabled: !!onToggleEnabled, onViewChunks: !!onViewChunks });
  return (
    <ScrollArea className="h-full">
      <div className="space-y-3 p-1">
        {files.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground border-2 border-dashed rounded-lg">
            <FileText className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p className="text-sm">No files attached</p>
          </div>
        ) : (
          <div className="space-y-2">
            {files.map((file, index) => (
              <div 
                key={index} 
                className="flex flex-col sm:flex-row items-start sm:items-center gap-2 sm:gap-3 p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
              >
                <div className="flex items-center gap-3 flex-1 min-w-0 w-full sm:w-auto">
                  <div className="shrink-0 w-10 h-10 rounded-md bg-primary/10 flex items-center justify-center">
                    <FileText className="h-5 w-5 text-primary" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <a 
                      href={file.localUrl} 
                      target="_blank" 
                      rel="noopener noreferrer" 
                      className="font-medium text-sm hover:underline block truncate"
                      title={file.name}
                    >
                      {file.name}
                    </a>
                    <div className="flex flex-wrap items-center gap-1 sm:gap-2 mt-0.5">
                      <span className="text-xs text-muted-foreground">{file.type || 'Unknown'}</span>
                      {file.chunkCount && (
                        <>
                          <span className="text-muted-foreground hidden sm:inline">•</span>
                          <span className="text-xs text-muted-foreground">{file.chunkCount} chunks</span>
                        </>
                      )}
                      {file.documentId && (
                        <>
                          <span className="text-muted-foreground hidden sm:inline">•</span>
                          <span className="text-xs font-mono text-muted-foreground" title={file.documentId}>
                            ID: {file.documentId.substring(0, 8)}
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                </div>

                <div className="shrink-0 w-full sm:w-auto">
                  <StatusBadge status={file.processingStatus || file.uploadStatus || file.downloadStatus} />
                  {file.statusMessage && (
                    <p className="text-xs text-muted-foreground mt-1 max-w-full sm:max-w-[150px] truncate" title={file.statusMessage}>
                      {file.statusMessage}
                    </p>
                  )}
                </div>

                <div className="flex flex-wrap items-center gap-1 shrink-0 w-full sm:w-auto">
                  {onToggleEnabled && file.processingStatus === 'done' && (
                    <div className="flex items-center gap-1.5 px-2">
                      <Switch 
                        checked={file.enabled !== false} 
                        onCheckedChange={(enabled) => onToggleEnabled(index, enabled)}
                        className="scale-90"
                      />
                      <span className="text-xs text-muted-foreground">
                        {file.enabled !== false ? 'On' : 'Off'}
                      </span>
                    </div>
                  )}
                  {onViewChunks && file.documentId && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        console.log('FileList Chunks clicked:', { 
                          onViewChunks: !!onViewChunks, 
                          documentId: file.documentId, 
                          name: file.name 
                        });
                        onViewChunks(file.documentId || '', file.name);
                      }}
                      className="gap-1.5 flex-1 sm:flex-none"
                    >
                      <BookOpen className="h-3.5 w-3.5" />
                      <span className="sm:inline">Chunks</span>
                    </Button>
                  )}
                  {previewFile && file.processingStatus === 'done' && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => previewFile(file)}
                      className="gap-1.5 flex-1 sm:flex-none"
                    >
                      <Eye className="h-3.5 w-3.5" />
                      <span className="sm:inline">Preview</span>
                    </Button>
                  )}
                  {onRemove && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => onRemove(index)}
                      className="text-destructive hover:text-destructive hover:bg-destructive/10"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </ScrollArea>
  );
}
