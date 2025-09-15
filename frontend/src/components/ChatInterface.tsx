"use client";

import { OLLAMA_MODELS } from "@/lib/ollamaModels";

import React, { useState, useRef, useEffect } from "react";
import {
  getAllChats,
  getChatById,
  addChat,
  updateChat,
  ChatDocument,
  MessageObject
} from "@/lib/chatStorage";

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
import { cloneChatFolderToLocal } from "@/lib/localFiles";

interface ChatInterfaceProps {
  activeDocument?: ProcessingResult;
}

export function ChatInterface({ activeDocument }: ChatInterfaceProps) {
  const [currentChatId, setCurrentChatId] = useState<string | undefined>();
  const [input, setInput] = useState("");
  const [files, setFiles] = useState<FileWithUrl[]>([]);
  const [model, setModel] = useState(OLLAMA_MODELS[0]);
  const [previewFile, setPreviewFile] = useState<FileWithUrl | string | null>(null);
  const [showAttachments, setShowAttachments] = useState(false);
  const [chats, setChats] = useState<ChatDocument[]>(() => getAllChats());
  const [isSidebarOpen, setIsSidebarOpen] = useState(() => {
    if (typeof window !== "undefined") {
      return window.innerWidth >= 640;
    }
    return true;
  });

  useEffect(() => {
    setChats(getAllChats());
  }, []);

  useEffect(() => {
    const run = async () => {
      if (!currentChatId) return;
      const chat = getChatById(currentChatId);
      if (!chat) return;
      const filesForChat = chat.fileWithUrl || [];
      if (filesForChat.length === 0) return;
      const mapping = await cloneChatFolderToLocal(currentChatId, filesForChat);
      const updated = getChatById(currentChatId);
      if (!updated) return;
      updated.fileWithUrl = (updated.fileWithUrl || []).map(f => ({ ...f, localUrl: mapping[f.name] || f.localUrl }));
      updated.message_objects = updated.message_objects.map(m => ({
        ...m,
        files: (m.files || []).map(f => ({ ...f, localUrl: mapping[f.name] || f.localUrl }))
      }));
      updateChat(updated);
      setChats(getAllChats());
    };
    run();
  }, [currentChatId]);

  useEffect(() => {
    const handleResize = () => {
      setIsSidebarOpen(window.innerWidth >= 640);
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

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

  const messages = currentChatId ? getChatById(currentChatId)?.message_objects || [] : [];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const fileArray = Array.from(e.target.files);
      let chatId = currentChatId;
      if (!chatId) {
        const newChatId = Math.random().toString(36).substring(7);
        const now = new Date().toISOString();
        const newChat: ChatDocument = {
          chat_id: newChatId,
          title: "New Chat",
          created_at: now,
          fileWithUrl: [],
          message_objects: [],
        };
        addChat(newChat);
        setChats(getAllChats());
        setCurrentChatId(newChatId);
        chatId = newChatId;
      }
      const chatFolder = `chats/${chatId}`;
      const uploaded: FileWithUrl[] = await Promise.all(
        fileArray.map(async (file) => {
          const { url, key, filename } = await uploadDocument(file, { chatFolder });
          return {
            name: filename || file.name,
            url,
            key,
            type: file.type,
            size: file.size,
            chatId,
          };
        })
      );
      try {
        const mapping = await cloneChatFolderToLocal(
          chatId,
          uploaded.map(f => ({ name: f.name, type: f.type, key: f.key }))
        );
        const withLocal = uploaded.map(f => ({ ...f, localUrl: mapping[f.name] || f.localUrl }));
        setFiles((prev) => [...prev, ...withLocal]);
      } catch {
        setFiles((prev) => [...prev, ...uploaded]);
      }
    }
  };

  const handleRemoveFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSidebarChatSelect = (chatId?: string) => {
    if (chatId) {
      setCurrentChatId(chatId);
      setInput("");
      setFiles([]);
    }
  };

  const handleSidebarNewChat = () => {
    const newChatId = Math.random().toString(36).substring(7);
    const now = new Date().toISOString();
    const newChat: ChatDocument = {
      chat_id: newChatId,
      title: "New Chat",
      created_at: now,
      fileWithUrl: [],
      message_objects: [],
    };
    addChat(newChat);
    setChats(getAllChats());
    setCurrentChatId(newChatId);
    setInput("");
    setFiles([]);
  };

  const handleSubmit = async () => {
    if (!input.trim() && files.length === 0) return;
    let chatId = currentChatId;
    let chat = chatId ? getChatById(chatId) : undefined;
    if (!chat) {
      // Create a new chat if it doesn't exist
      const newChatId = Math.random().toString(36).substring(7);
      const now = new Date().toISOString();
      chat = {
        chat_id: newChatId,
        title: "New Chat",
        created_at: now,
        fileWithUrl: [],
        message_objects: [],
      };
      addChat(chat);
      setChats(getAllChats());
      setCurrentChatId(newChatId);
      chatId = newChatId;
    }
    const messageId = Math.random().toString(36).substring(7);
    const now = new Date().toISOString();
    const currentInput = input.trim();
    const currentFiles = [...files];

    // Cloned-file workflow: no remote URLs in prompt
    const promptWithFiles = currentInput;

    const userMsg: MessageObject = {
      message_id: messageId,
      author: "user",
      content: currentInput,
      created_at: now,
      files: currentFiles,
    };
    chat.message_objects.push(userMsg);
    // Keep chat-level file list in sync (unique by key+name)
    const existing = chat.fileWithUrl || [];
    const seen = new Set(existing.map(f => `${f.key || ''}|${f.name}`));
    for (const f of currentFiles) {
      const uniq = `${f.key || ''}|${f.name}`;
      if (!seen.has(uniq)) {
        existing.push(f);
        seen.add(uniq);
      }
    }
    chat.fileWithUrl = existing;
    updateChat(chat);
    setChats(getAllChats());
    setInput("");
    setFiles([]);

    try {
      if (currentInput.trim()) {
        const botMsgId = Math.random().toString(36).substring(7);
        const botMsg: MessageObject = {
          message_id: botMsgId,
          author: "ai",
          content: "",
          created_at: new Date().toISOString(),
          files: [],
        };
        chat.message_objects.push(botMsg);
        updateChat(chat);
        setChats(getAllChats());
        if (stream) {
          let fullResponse = "";
          await sendExternalChatMessage({
            prompt: promptWithFiles,
            model,
            stream: true,
            onStreamChunk: (chunk) => {
              fullResponse += chunk;
              // Update bot message in chat
              const updatedChat = getChatById(chatId!);
              if (!updatedChat) return;
              const msgIdx = updatedChat.message_objects.findIndex(m => m.message_id === botMsgId);
              if (msgIdx !== -1) {
                updatedChat.message_objects[msgIdx].content = fullResponse;
                updateChat(updatedChat);
                setChats(getAllChats());
              }
            },
          });
        } else {
          const chatResponse = await sendExternalChatMessage({
            prompt: promptWithFiles,
            model,
            stream: false,
          });
          // Update bot message in chat
          const updatedChat = getChatById(chatId!);
          if (updatedChat) {
            const msgIdx = updatedChat.message_objects.findIndex(m => m.message_id === botMsgId);
            if (msgIdx !== -1) {
              updatedChat.message_objects[msgIdx].content = chatResponse.response;
              updateChat(updatedChat);
              setChats(getAllChats());
            }
          }
        }
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Something went wrong");
      // Optionally remove last user message on error
    }
  };

  const getAllAttachedFiles = () => {
    if (!currentChatId) return [];
    const chat = getChatById(currentChatId);
    if (!chat) return [];
    return chat.message_objects.reduce((acc: FileWithUrl[], msg) => {
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
        chats={chats}
        currentChatId={currentChatId}
        onChatSelect={handleSidebarChatSelect}
        onNewChatStart={handleSidebarNewChat}
        onChatsUpdate={() => setChats(getAllChats())}
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
                {!currentChatId || !getChatById(currentChatId) || getChatById(currentChatId)!.message_objects.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-[calc(100vh-10rem)] p-4">
                    <div className="max-w-2xl w-full space-y-6">
                      <div className="text-center space-y-2">
                        <MessageSquare className="h-12 w-12 text-primary mx-auto mb-2" />
                        <h2 className="text-2xl font-semibold">
                          Start a New Conversation
                        </h2>
                        <p className="text-sm text-muted-foreground">
                          Upload documents or start typing to begin chatting with AI.
                        </p>
                      </div>
                    </div>
                  </div>
                ) : (
                  messages.map((message, idx, arr) => {
                    const isLast = idx === arr.length - 1;
                    const showPulse =
                      stream &&
                      isLast &&
                      message.author === "ai" &&
                      !message.content;
                    return (
                      <div
                        key={message.message_id}
                        className={`flex items-start gap-3 w-full ${
                          message.author === "user"
                            ? "flex-row-reverse"
                            : "flex-row"
                        }`}
                      >
                        {/* Avatar */}
                        {message.author === "user" ? (
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
                                {(user?.fullName || user?.username || "U")[0].toUpperCase()}
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
                              message.author === "user"
                                ? "bg-primary text-primary-foreground ml-auto"
                                : "bg-muted"
                            )}
                            style={{ wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}
                          >
                            <div className="space-y-2 overflow-hidden">
                              {(message.files && message.files.length > 0) && (
                                <div className="flex flex-wrap gap-2 mb-2">
                                  {message.files.map((file, index) => (
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
                                ) : message.author === "ai" ? (
                                  <MarkdownRenderer content={message.content} />
                                ) : (
                                  message.content
                                )}
                              </div>
                            </div>
                          </Card>
                          <div
                            className={`flex gap-2 ${
                              message.author === "user"
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
                            {message.author === "ai" && (
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
              files={(currentChatId && getChatById(currentChatId)?.fileWithUrl?.length)
                ? (getChatById(currentChatId)!.fileWithUrl)
                : getAllAttachedFiles()}
              onRemove={() => {}}
              previewFile={setPreviewFile}
            />
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
