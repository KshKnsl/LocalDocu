'use client';

import { cn } from '@/lib/utils';
import { FileCard } from './FileCard';

import type { FileWithUrl } from "./FileWithUrl";


interface FileListProps {
  files: FileWithUrl[];
  onRemove: (index: number) => void;
  previewFile?: (file: FileWithUrl) => void;
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
        <FileCard
          key={index}
          file={file}
          onPreview={previewFile}
          onRemove={() => onRemove(index)}
        />
      ))}
    </div>
  );
}
