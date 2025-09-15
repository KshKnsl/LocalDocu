'use client';

import React, { useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ExternalLink, FileText, X, FileQuestion, Download } from 'lucide-react';
import { cn } from '@/lib/utils';
import {
  Collapsible,
  CollapsibleContent,
} from "@/components/ui/collapsible";
import { ScrollArea } from '@radix-ui/react-scroll-area';

import type { FileWithUrl } from "./ui/FileWithUrl";

interface FilePreviewProps {
  file: FileWithUrl | string | null;
  onClose: () => void;
  className?: string;
}

type FileType = 'pdf' | 'text' | 'image' | 'other';

function getFileType(input: FileWithUrl | string): FileType {
  const fileName = typeof input === 'string' ? input : input.name;
  const extension = fileName.split('.').pop()?.toLowerCase();
  if (!extension) return 'other';
  if ([
    'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'svg', 'ico'
  ].includes(extension)) return 'image';
  if (extension === 'pdf') return 'pdf';
  if ([
    'txt', 'md', 'csv', 'log',
    'js', 'jsx', 'ts', 'tsx', 'py', 'java', 'cpp', 'c', 'h', 'hpp', 'cs', 'php', 'rb',
    'html', 'css', 'scss', 'json', 'xml',
    'yaml', 'yml', 'ini', 'conf', 'toml'
  ].includes(extension)) return 'text';
  return 'other';
}
export function FilePreview({ file, onClose, className }: FilePreviewProps) {
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isOpen, setIsOpen] = useState(true);

  useEffect(() => {
    if (!file) {
      setContent('');
      setLoading(false);
      return;
    }

    const loadFileContent = async () => {
      try {
        setLoading(true);
        setError(null);
        if (typeof file === 'string') {
          setContent(file);
        } else if (file.url) {
          // For text files, fetch content; otherwise, just use the URL
          if (getFileType(file) === 'text') {
            const res = await fetch(file.url);
            const text = await res.text();
            setContent(text);
          } else {
            setContent(file.url);
          }
        } else {
          throw new Error('Invalid file input');
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Error loading file preview';
        setError(errorMessage);
        console.error('Error loading file:', err);
      } finally {
        setLoading(false);
      }
    };

    loadFileContent();
  }, [file]);

  const renderPreview = () => {
    if (!file || loading || error) {
      return (
        <div className="flex flex-col items-center justify-center gap-4 p-8 text-muted-foreground h-full">
          <FileQuestion className="w-12 h-12" />
          <p className="text-center">{error || 'No file selected'}</p>
          <p className="text-sm text-center text-muted-foreground">
            {!error && 'Choose a file to preview its contents here'}
          </p>
        </div>
      );
    }

  const fileType = getFileType(file);
  const FileIcon = FileText;
  const fileName = typeof file === 'string' ? file.split('/').pop() : file.name;

    switch (fileType) {
      case 'pdf': {
        const pdfUrl = typeof file === 'string' ? file : file.url;
        return (
          <div className="w-full h-[calc(100vh-10rem)] bg-white rounded-lg overflow-hidden">
            <object
              data={pdfUrl}
              type="application/pdf"
              className="w-full h-full"
            >
              <div className="flex flex-col items-center justify-center gap-4 p-8">
                <p className="text-center">PDF preview not available</p>
                <p className="text-sm text-muted-foreground">Please use the open button above to view this PDF</p>
              </div>
            </object>
          </div>
        );
      }
      case 'text':
        return (
          <Card className="bg-muted/50 h-full">
            <ScrollArea className="h-[calc(100vh-10rem)] w-full">
              <pre className="p-4 text-sm font-mono">
                <code>{content}</code>
              </pre>
            </ScrollArea>
          </Card>
        );
      case 'image': {
        const imgSrc = typeof file === 'string' ? file : file.url;
        if (!imgSrc) return null;
        return (
          <div className="w-full h-[calc(100vh-10rem)] bg-white/5 dark:bg-black/20 rounded-lg overflow-auto">
            <div className="flex items-center justify-center min-h-full p-4">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={imgSrc}
                alt={fileName || 'Image preview'}
                className="max-w-full max-h-full object-contain rounded-lg"
                loading="lazy"
              />
            </div>
          </div>
        );
      }
      default:
        return (
          <div className="flex flex-col items-center justify-center gap-4 p-8 text-muted-foreground">
            <FileIcon className="w-12 h-12" />
            <p>This file type cannot be previewed</p>
            <p className="text-sm">{fileName}</p>
          </div>
        );
    }
  };

  return (
    <Collapsible
      open={isOpen}
      onOpenChange={setIsOpen}
      className={cn(
        "relative bg-background border-l data-[state=open]:w-[350px] data-[state=closed]:w-[50px] transition-all duration-300",
        className
      )}
    >
      <div className="sticky top-0 z-10 border-b bg-muted/50">
        <div className="flex items-center justify-between gap-2 p-2">
          <div className="flex items-center flex-1 min-w-0">
            <span className="text-sm font-medium truncate">
              {typeof file === 'string' ? file.split('/').pop() : file?.name || 'No file selected'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            {file && (
              <>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => {
                    const url = typeof file === 'string' ? file : file.url;
                    window.open(url, '_blank');
                  }}
                  className="h-7 w-7"
                  title="Open in new tab"
                >
                  <ExternalLink className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => {
                    const url = typeof file === 'string' ? file : file.url;
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = (typeof file === 'string' ? file.split('/').pop() : file.name) || 'download';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                  }}
                  className="h-7 w-7"
                  title="Download"
                >
                  <Download className="h-4 w-4" />
                </Button>
              </>
            )}
            <Button
              variant="ghost"
              size="icon"
              onClick={onClose}
              className="h-7 w-7"
              title="Close preview"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      <CollapsibleContent className="overflow-auto p-4 max-h-[calc(100vh-4rem)]">
        {renderPreview()}
      </CollapsibleContent>
    </Collapsible>
  );
}
