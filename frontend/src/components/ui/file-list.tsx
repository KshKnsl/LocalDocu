'use client';

import { Button } from './button';
import { cn } from '@/lib/utils';
import { FileText, X, Eye } from 'lucide-react';

interface FileListProps {
  files: File[];
  onRemove: (index: number) => void;
  previewFile?: (file: File) => void;
}

export function FileList({ files, onRemove, previewFile }: FileListProps) {
  return (
    <div
      className={cn(
        "space-y-2",
        files.length > 4 &&
          "max-h-[200px] overflow-y-auto pr-2 [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-muted-foreground/20 [&::-webkit-scrollbar-track]:rounded-full [&::-webkit-scrollbar-track]:bg-muted"
      )}
    >
      {files.map((file, index) => (
        <div
          key={index}
          className="flex items-center gap-2 p-2 pr-3 bg-muted/50 rounded-lg text-sm"
        >
          <FileText className="w-4 h-4 text-muted-foreground" />
          <span className="flex-1 truncate">{file.name}</span>
          <div className="flex gap-1">
            {previewFile && (
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6"
                onClick={() => previewFile(file)}
              >
                <Eye className="h-4 w-4" />
              </Button>
            )}
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={() => onRemove(index)}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      ))}
    </div>
  );
}
