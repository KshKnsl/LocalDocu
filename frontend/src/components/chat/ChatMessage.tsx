"use client";

import React from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Bot, Copy, Link2 } from "lucide-react";
import { FileCard } from "@/components/ui/FileCard";
import { MarkdownRenderer } from "@/components/MarkdownRenderer";
import { CitationCard } from "@/components/CitationCard";
import { PulseLoader } from "@/components/ui/pulse";
import type { FileWithUrl } from "@/components/ui/FileWithUrl";
import { MessageObject } from "@/lib/chatStorage";
import { cn } from "@/lib/utils";
import { copyMessage, copyCurrentUrl } from "@/lib/clipboard";

interface ChatMessageProps {
  message: MessageObject;
  isLast: boolean;
  userImageUrl?: string;
  userName?: string;
  modelDisplayName?: string;
  onPreviewFile: (file: FileWithUrl) => void;
}

export function ChatMessage({
  message,
  isLast,
  userImageUrl,
  userName,
  modelDisplayName,
  onPreviewFile,
}: ChatMessageProps) {
  const handleCopyMessage = () => copyMessage(message.content);
  const handleCopyLink = () => copyCurrentUrl();

  return (
    <div
      className={`flex items-start gap-3 w-full ${
        message.author === "user" ? "flex-row-reverse" : "flex-row"
      }`}
    >
      {message.author === "user" ? (
        userImageUrl ? (
          /* eslint-disable-next-line @next/next/no-img-element */
          <img
            src={userImageUrl}
            alt={userName || "User"}
            className="h-8 w-8 rounded-full object-cover"
          />
        ) : (
          <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
            <span className="text-sm font-medium">
              {(userName || "U")[0].toUpperCase()}
            </span>
          </div>
        )
      ) : (
        <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
          <Bot className="h-5 w-5 text-primary" />
        </div>
      )}

      <div className={`flex flex-col gap-2 max-w-[75%] overflow-hidden`}>
        <Card
          className={cn(
            "p-4 w-full max-w-3xl overflow-x-auto scrollbar-thin scrollbar-thumb-muted-foreground/30",
            message.author === "user"
              ? "bg-primary text-primary-foreground ml-auto"
              : "bg-muted"
          )}
          style={{ wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}
        >
          <div className="space-y-2 overflow-hidden">
            {message.author === "ai" && modelDisplayName && (
              <div className="flex items-center gap-2 mb-1">
                <span className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20 font-medium">
                  {modelDisplayName}
                </span>
              </div>
            )}
            {message.files && message.files.length > 0 && (
              <div className="flex flex-wrap gap-2 mb-2">
                {message.files.map((file, index) => (
                  <FileCard
                    key={`file-${index}`}
                    file={file}
                    onPreview={onPreviewFile}
                  />
                ))}
              </div>
            )}
            <div
              className="overflow-x-auto max-w-full"
              style={{ wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}
            >
              {message.author === "ai" && message.content.startsWith("__STATUS__:") ? (
                <span className="flex items-center gap-2">
                  <PulseLoader />
                  <span className="text-sm text-muted-foreground">
                    {message.content.replace("__STATUS__:", "")}
                  </span>
                </span>
              ) : message.author === "ai" ? (
                <MarkdownRenderer content={message.content} />
              ) : (
                message.content
              )}
            </div>
            {message.author === "ai" && message.citations && message.citations.length > 0 && (
              <div className="mt-3 pt-3 border-t border-border/40">
                <div className="text-xs text-muted-foreground mb-2">Sources used:</div>
                <div className="flex flex-wrap gap-2">
                  {message.citations.map((citation, idx) => (
                    <CitationCard key={idx} citation={citation} />
                  ))}
                </div>
              </div>
            )}
          </div>
        </Card>
        <div
          className={`flex gap-2 ${
            message.author === "user" ? "justify-end" : "justify-start"
          }`}
        >
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 text-muted-foreground hover:text-foreground"
            onClick={handleCopyMessage}
          >
            <Copy className="h-4 w-4" />
          </Button>
          {message.author === "ai" && (
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 text-muted-foreground hover:text-foreground"
              onClick={handleCopyLink}
            >
              <Link2 className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
