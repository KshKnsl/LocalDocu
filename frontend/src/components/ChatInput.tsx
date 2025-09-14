"use client";


import React, { useRef } from "react";
import { OLLAMA_MODELS } from "@/lib/ollamaModels";
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Paperclip, Send, Link2 } from "lucide-react";
import { FileCard } from "@/components/ui/FileCard";

interface ChatInputProps {
  input: string;
  setInput: (value: string) => void;
  files: File[];
  onFileSelect: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onRemoveFile: (index: number) => void;
  onSubmit: () => void;
  onPreviewFile: (file: File) => void;
  onShowAttachments: () => void;
  model: string;
  setModel: (model: string) => void;
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
  model,
  setModel,
}: ChatInputProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  return (
    <div className="border-t bg-background p-2 w-full">
      <div className="mx-auto flex gap-2 items-center mb-2">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Model:</span>
          <Select value={model} onValueChange={setModel}>
            <SelectTrigger className="w-[140px] h-8">
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent>
              {OLLAMA_MODELS.map((m) => (
                <SelectItem key={m} value={m}>
                  {m}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="border rounded-lg bg-muted/30 px-3 py-2 flex items-center w-full overflow-hidden">
          <div className="text-sm font-medium flex items-center gap-2 shrink-0">
            <span className="text-xs text-muted-foreground">
              {files.length} {files.length === 1 ? "file" : "files"}
            </span>
          </div>
          <div className="flex-1 min-w-0 mx-2 overflow-x-auto [&::-webkit-scrollbar]:hidden">
            <div className="flex gap-2 w-fit min-w-full">
              {files.map((file, index) => (
                <FileCard
                  key={index}
                  file={file}
                  onPreview={onPreviewFile}
                  onRemove={() => onRemoveFile(index)}
                />
              ))}
              {files.length === 0 && (
                <div className="text-sm text-muted-foreground py-0.5">
                  No files attached
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      <div className="mx-auto flex gap-2 items-end">
        <input
          type="file"
          ref={fileInputRef}
          className="hidden"
          onChange={onFileSelect}
          multiple
        />
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              onSubmit();
            }
          }}
          className="flex-1 min-h-[80px] max-h-[160px] rounded-md border bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 resize-none overflow-y-auto [&::-webkit-scrollbar]:hidden"
        />
        <div className="flex flex-col gap-2">
          <Button
            onClick={onSubmit}
            disabled={!input.trim() && files.length === 0}
            className="shrink-0 px-4 w-full"
          >
            <Send className="h-4 w-4 mr-2" />
            <span>Send</span>
          </Button>
          <div className="flex gap-2 justify-center">
            <Button
              variant="outline"
              size="icon"
              onClick={() => fileInputRef.current?.click()}
              className="shrink-0"
              title="Attach files"
            >
              <Paperclip className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              onClick={onShowAttachments}
              className="shrink-0"
              title="View attachments"
            >
              <Link2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
