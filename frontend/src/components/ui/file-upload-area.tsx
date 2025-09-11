'use client';

import { Input } from './input';
import { cn } from '@/lib/utils';
import { FileText, Upload } from 'lucide-react';

interface FileUploadAreaProps {
  files: File[];
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  loading?: boolean;
  accept?: string;
  id: string;
}

export function FileUploadArea({
  files,
  onChange,
  loading,
  accept,
  id
}: FileUploadAreaProps) {
  return (
    <div
      className={cn(
        "relative border-2 border-dashed rounded-xl transition-colors",
        files.length > 0
          ? "border-primary bg-primary/5"
          : "border-muted-foreground/25 hover:border-primary/50",
        loading ? "opacity-50 pointer-events-none" : ""
      )}
    >
      <Input
        type="file"
        accept={accept}
        onChange={onChange}
        className="hidden"
        multiple
        id={id}
      />
      <label
        htmlFor={id}
        className="block p-8 cursor-pointer"
      >
        <div className="flex flex-col items-center gap-3">
          <div className="p-3 rounded-full bg-primary/10">
            {files.length > 0 ? (
              <FileText className="w-6 h-6 text-primary" />
            ) : (
              <Upload className="w-6 h-6 text-primary" />
            )}
          </div>
          <div className="text-center">
            <p className="font-medium text-primary">
              {files.length > 0 ? "Change files" : "Upload your documents"}
            </p>
            <p className="text-sm text-muted-foreground mt-1">
              PDF, Word, or text files
            </p>
          </div>
        </div>
      </label>
    </div>
  );
}
