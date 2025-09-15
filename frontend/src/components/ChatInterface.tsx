"use client";

import { OLLAMA_MODELS } from "@/lib/ollamaModels";

import React, { useState, useRef, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { FilePreview } from "./FilePreview";
import { cn } from "@/lib/utils";
import { Copy, Link2, MessageSquare, Bot } from "lucide-react";
import { MarkdownRenderer } from "./MarkdownRenderer";
import { FileCard } from "@/components/ui/FileCard";
import type { FileWithUrl } from "@/components/ui/FileWithUrl";
import {
  ProcessingResult,
  sendExternalChatMessage,
  uploadDocument,
  processDocument,
} from "@/lib/api";
import { useUser } from "@clerk/nextjs";
import { NewChatFileUpload } from "./NewChatFileUpload";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ChatSidebar } from "./ChatSidebar";
import { ChatInput } from "./ChatInput";
import { FileList } from "./ui/file-list";

interface Message {
  id: string;
  type: "user" | "bot";
  content: string;
  timestamp: Date;
  files?: FileWithUrl[];
  processedFiles?: string[];
}

// Chat interface moved to ChatSidebar component where it's used

interface ChatInterfaceProps {
  activeDocument?: ProcessingResult;
}

export function ChatInterface({ activeDocument }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [model, setModel] = useState(OLLAMA_MODELS[0]);
  const [input, setInput] = useState("");
  const [files, setFiles] = useState<FileWithUrl[]>([]);
  const [previewFile, setPreviewFile] = useState<FileWithUrl | string | null>(null);
  const [showAttachments, setShowAttachments] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(() => {
    if (typeof window !== "undefined") {
      return window.innerWidth >= 640; // open by default on desktop, collapsed on small screens
    }
    return true;
  });

  useEffect(() => {
    const handleResize = () => {
      setIsSidebarOpen(window.innerWidth >= 640);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const [currentChatId, setCurrentChatId] = useState<string | undefined>();
  const [chatMessages, setChatMessages] = useState<Record<string, Message[]>>(
    {}
  );

  const [stream, setStreamState] = useState(() => {
    if (typeof window !== "undefined") {
      const stored = localStorage.getItem("chat_stream_toggle");
      return stored === null ? false : stored === "true";
    }
    return false;
  });
  useEffect(() => {
    localStorage.setItem("chat_stream_toggle", String(stream));
  }, [stream]);
  const setStream = (v: boolean) => setStreamState(v);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { user } = useUser();

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const fileArray = Array.from(e.target.files);
      // Upload all files and get their S3 URLs
      const uploaded: FileWithUrl[] = await Promise.all(
        fileArray.map(async (file) => {
          const { url } = await uploadDocument(file);
          return {
            name: file.name,
            url,
            type: file.type,
            size: file.size,
          };
        })
      );
      setFiles((prev) => [...prev, ...uploaded]);
    }
  };

  const handleRemoveFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSidebarChatSelect = (chatId: string) => {
    setCurrentChatId(chatId);
    setMessages(chatMessages[chatId] || []);
  };

  const handleSidebarNewChat = () => {
    setCurrentChatId(undefined);
    setMessages([]);
    setInput("");
    setFiles([]);
  };

  useEffect(() => {
    if (currentChatId) {
      setChatMessages((prev) => ({
        ...prev,
        [currentChatId]: messages,
      }));
    }
  }, [messages, currentChatId]);

  const handleSubmit = async () => {
    if (!input.trim() && files.length === 0) return;
    const messageId = Math.random().toString(36).substring(7);
    const currentInput = input.trim();
  const currentFiles = [...files];

    if (currentInput || currentFiles.length > 0) {
      setMessages((prev) => [
        ...prev,
        {
          id: messageId,
          type: "user",
          content: currentInput,
          files: currentFiles,
          processedFiles: [],
          timestamp: new Date(),
        },
      ]);
    }

    setInput("");
    setFiles([]);

    try {

      if (currentInput.trim()) {
        const botMsgId = Math.random().toString(36).substring(7);
        if (stream) {
          // Streaming chat response
          setMessages((prev) => [
            ...prev,
            {
              id: botMsgId,
              type: "bot",
              content: "",
              timestamp: new Date(),
            },
          ]);
          let fullResponse = "";
          await sendExternalChatMessage({
            prompt: currentInput,
            model,
            stream: true,
            onStreamChunk: (chunk) => {
              fullResponse += chunk;
              setMessages((prev) =>
                prev.map((msg) =>
                  msg.id === botMsgId ? { ...msg, content: fullResponse } : msg
                )
              );
            },
          });
        } else {
          // Non-streaming chat response
          const chatResponse = await sendExternalChatMessage({
            prompt: currentInput,
            model,
            stream: false,
          });
          setMessages((prev) => [
            ...prev,
            {
              id: botMsgId,
              type: "bot",
              content: chatResponse.response,
              timestamp: new Date(),
            },
          ]);
        }
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Something went wrong");
      if (currentInput) setMessages((prev) => prev.slice(0, -1));
    }
  };

  const getAllAttachedFiles = () => {
    return messages.reduce((acc: FileWithUrl[], msg) => {
      if (msg.files) {
        acc.push(...msg.files);
      }
      return acc;
    }, []);
  };
  return (
    <div className="flex w-full h-full">
      <ChatSidebar
        isSidebarOpen={isSidebarOpen}
        setIsSidebarOpen={setIsSidebarOpen}
        onChatSelect={handleSidebarChatSelect}
        onNewChatStart={handleSidebarNewChat}
        stream={stream}
        setStream={setStream}
      />

      <main className="flex-1 flex flex-col h-screen overflow-hidden">
        {/* Main chat section */}
        <div className="flex-1 flex flex-col min-h-0">
          {/* Chat messages area */}
          <div className="h-[calc(100vh-10rem)] w-full pt-5">
            <ScrollArea className="h-full w-full">
              <div className="space-y-6 p-4 pt-5 max-w-full">
                {messages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-[calc(100vh-10rem)] p-4">
                    <div className="max-w-2xl w-full space-y-6">
                      <div className="text-center space-y-2">
                        <MessageSquare className="h-12 w-12 text-primary mx-auto mb-2" />
                        <h2 className="text-2xl font-semibold">
                          Start a New Conversation
                        </h2>
                        <p className="text-sm text-muted-foreground">
                          Upload documents to analyze or ask questions directly
                        </p>
                      </div>

                      <NewChatFileUpload
                        onProcessingComplete={(result) => {
                          setMessages((prev) => [
                            ...prev,
                            {
                              id: Math.random().toString(36).substring(7),
                              type: "bot",
                              content: `Document processed successfully! You can now ask questions about "${result.documentId}". The document has been split into ${result.chunkCount} chunks for efficient processing.`,
                              timestamp: new Date(),
                            },
                          ]);
                        }}
                      />

                      <div className="relative">
                        <div className="absolute inset-0 flex items-center">
                          <div className="w-full border-t"></div>
                        </div>
                        <div className="relative flex justify-center text-xs uppercase">
                          <span className="bg-background px-2 text-muted-foreground">
                            or
                          </span>
                        </div>
                      </div>

                      <div className="text-center space-y-2">
                        <p className="text-sm font-medium">
                          Start typing your question
                        </p>
                        <p className="text-xs text-muted-foreground">
                          The AI will assist you based on your query
                        </p>
                      </div>
                    </div>
                  </div>
                ) : (
                  messages.map((message, idx) => {
                    // Pulse animation if streaming, last message, bot, and content is empty
                    const isLast = idx === messages.length - 1;
                    const showPulse =
                      stream &&
                      isLast &&
                      message.type === "bot" &&
                      !message.content;
                    return (
                      <div
                        key={message.id}
                        className={`flex items-start gap-3 w-full ${
                          message.type === "user"
                            ? "flex-row-reverse"
                            : "flex-row"
                        }`}
                      >
                        {/* Avatar */}
                        {message.type === "user" ? (
                          user?.imageUrl ? (
                            /* eslint-disable-next-line @next/next/no-img-element */
                            <img
                              src={user.imageUrl}
                              alt={user.fullName || user.username || "User"}
                              className="h-8 w-8 rounded-full object-cover"
                            />
                          ) : (
                            <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
                              <span className="text-sm font-medium">
                                {(user?.fullName ||
                                  user?.username ||
                                  "U")[0].toUpperCase()}
                              </span>
                            </div>
                          )
                        ) : (
                          <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
                            <Bot className="h-5 w-5 text-primary" />
                          </div>
                        )}

                        {/* Message Content */}
                        <div
                          className={`flex flex-col gap-2 max-w-[75%] overflow-hidden`}
                        >
                          <Card
                            className={cn(
                              "p-4 w-full max-w-3xl overflow-x-auto scrollbar-thin scrollbar-thumb-muted-foreground/30",
                              message.type === "user"
                                ? "bg-primary text-primary-foreground ml-auto"
                                : "bg-muted"
                            )}
                            style={{ wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}
                          >
                            <div className="space-y-2 overflow-hidden">
                              {((message.processedFiles &&
                                message.processedFiles.length > 0) ||
                                (message.files && message.files.length > 0)) && (
                                <div className="flex flex-wrap gap-2 mb-2">
                                  {message.files?.map((file, index) => (
                                    <FileCard
                                      key={`file-${index}`}
                                      file={file}
                                      onPreview={setPreviewFile}
                                    />
                                  ))}
                                </div>
                              )}
                              <div
                                className="overflow-x-auto max-w-full"
                                style={{ wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}
                              >
                                {showPulse ? (
                                  <span className="flex items-center gap-2">
                                    <span className="text-xs text-muted-foreground">Waiting for response</span>
                                    {/* @ts-ignore */}
                                    {React.createElement(require("./ui/pulse").PulseLoader)}
                                  </span>
                                ) : message.type === "bot" ? (
                                  <MarkdownRenderer content={message.content} />
                                ) : (
                                  message.content
                                )}
                              </div>
                            </div>
                          </Card>
                          <div
                            className={`flex gap-2 ${
                              message.type === "user"
                                ? "justify-end"
                                : "justify-start"
                            }`}
                          >
                            <Button
                              variant="ghost"
                              size="icon"
                              className="h-6 w-6 text-muted-foreground hover:text-foreground"
                              onClick={async () => {
                                await navigator.clipboard.writeText(
                                  message.content
                                );
                                toast.success("Message copied to clipboard", {
                                  duration: 2000,
                                  position: "bottom-right",
                                });
                              }}
                            >
                              <Copy className="h-4 w-4" />
                            </Button>
                            {message.type === "bot" && (
                              <Button
                                variant="ghost"
                                size="icon"
                                className="h-6 w-6 text-muted-foreground hover:text-foreground"
                                onClick={async () => {
                                  await navigator.clipboard.writeText(
                                    window.location.href
                                  );
                                  toast.success("Link copied to clipboard", {
                                    duration: 2000,
                                    position: "bottom-right",
                                  });
                                }}
                              >
                                <Link2 className="h-4 w-4" />
                              </Button>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })
                )}
                <div ref={messagesEndRef} />
              </div>
            </ScrollArea>
          </div>
        </div>

        {/* Chat Input */}
        <ChatInput
          input={input}
          setInput={setInput}
          files={files}
          onFileSelect={handleFileSelect}
          onRemoveFile={handleRemoveFile}
          onSubmit={handleSubmit}
          onPreviewFile={setPreviewFile}
          onShowAttachments={() => setShowAttachments(true)}
          model={model}
          setModel={setModel}
        />
      </main>

      {previewFile && (
        <FilePreview
          file={previewFile}
          onClose={() => setPreviewFile(null)}
        />
      )}

      <Dialog open={showAttachments} onOpenChange={setShowAttachments}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Attached Files</DialogTitle>
          </DialogHeader>
          <div className="max-h-[60vh] overflow-y-auto px-1">
            <FileList
              files={getAllAttachedFiles()}
              onRemove={() => {}}
              previewFile={setPreviewFile}
            />
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
