"use client";

import React, { useRef } from "react";
import { Button } from "@/components/ui/button";
import { Eye, Paperclip, Send, X, Link2 } from "lucide-react";

interface ChatInputProps {
  input: string;
  setInput: (value: string) => void;
  files: File[];
  onFileSelect: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onRemoveFile: (index: number) => void;
  onSubmit: () => void;
  onPreviewFile: (file: File) => void;
  onShowAttachments: () => void;
  loading: boolean;
}

export function ChatInput({
  input,
  setInput,
  files,
  onFileSelect,
  onRemoveFile,
  onSubmit,
  onPreviewFile,
  onShowAttachments,
  loading,
}: ChatInputProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  return (
    <div className="border-t bg-background p-2 w-full">
      <div className="mx-auto grid grid-cols-[300px,1fr] gap-2">
        <div className="border rounded-lg bg-muted/30 px-3 py-2 flex items-center w-full overflow-hidden">
          <div className="text-sm font-medium flex items-center gap-2 shrink-0">
            <span className="text-xs text-muted-foreground">
              {files.length} {files.length === 1 ? "file" : "files"}
            </span>
          </div>
          <div className="flex-1 min-w-0 mx-2 overflow-x-auto [&::-webkit-scrollbar]:hidden">
            <div className="flex gap-2 w-fit min-w-full">
              {files.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center gap-1.5 bg-background/50 border rounded-lg px-2 py-1 text-sm shrink-0 hover:bg-accent/50 transition-colors"
                >
                  <span className="truncate max-w-[100px]">{file.name}</span>
                  <div className="flex items-center">
                    <button
                      onClick={() => onPreviewFile(file)}
                      className="p-1 opacity-60 hover:opacity-100 rounded-sm hover:bg-background"
                    >
                      <Eye className="h-3.5 w-3.5" />
                    </button>
                    <button
                      onClick={() => onRemoveFile(index)}
                      className="p-1 text-destructive/60 hover:text-destructive rounded-sm hover:bg-background"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              ))}
              {files.length === 0 && (
                <div className="text-sm text-muted-foreground py-0.5">
                  No files attached
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Input column */}
        <div className="flex gap-2">
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            onChange={onFileSelect}
            multiple
            disabled={loading}
          />

          <div className="flex-1 flex flex-col">
            <div className="flex gap-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={
                  loading ? "Waiting for response..." : "Type your message..."
                }
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !loading) {
                    e.preventDefault();
                    onSubmit();
                  }
                }}
                disabled={loading}
                className="flex-1 min-h-[80px] max-h-[160px] rounded-md border bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 resize-none overflow-y-auto [&::-webkit-scrollbar]:hidden"
              />
              <div className="flex flex-col gap-2">
                <Button
                  onClick={onSubmit}
                  disabled={loading || (!input.trim() && files.length === 0)}
                  className="shrink-0 px-4 w-full"
                >
                  {loading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin mr-2" />
                      <span>Sending...</span>
                    </>
                  ) : (
                    <>
                      <Send className="h-4 w-4 mr-2" />
                      <span>Send</span>
                    </>
                  )}
                </Button>
                <div className="flex gap-2 justify-center">
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={loading}
                    className="shrink-0"
                    title="Attach files"
                  >
                    <Paperclip className="h-4 w-4" />
                  </Button>
                  <Button
                    variant="outline"
                    size="icon"
                    onClick={onShowAttachments}
                    disabled={loading}
                    className="shrink-0"
                    title="View attachments"
                  >
                    <Link2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
