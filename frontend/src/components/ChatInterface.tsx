"use client";
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
import { PulseLoader } from "./ui/pulse";
import { CitationCard } from "./CitationCard";
import { LOCAL_MODELS } from "@/lib/localModels";
import {
  ProcessingResult,
  sendExternalChatMessage,
  uploadDocument,
  processDocument,
  isUsingCustomBackend,
  getProgress,
  clearProgress,
  ProgressData,
} from "@/lib/api";
import { useUser } from "@clerk/nextjs";
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
import { BackendConfigDialog } from "./BackendConfig";
import { ProcessingBanner } from "./ProcessingBanner";
import { ChunkViewer } from "./ChunkViewer";

interface ChatInterfaceProps {
  activeDocument?: ProcessingResult;
}

interface ProcessingFile {
  name: string;
  status: 'uploading' | 'processing' | 'done' | 'failed';
  progress?: number;
  chunks?: number;
  currentChunk?: number;
}

export function ChatInterface({ activeDocument }: ChatInterfaceProps) {
  const [currentChatId, setCurrentChatId] = useState<string | undefined>();
  const [input, setInput] = useState("");
  const [files, setFiles] = useState<FileWithUrl[]>([]);
  const [model, setModel] = useState("remote"); // Default to remote
  const [previewFile, setPreviewFile] = useState<FileWithUrl | string | null>(null);
  const [showAttachments, setShowAttachments] = useState(false);
  const [chats, setChats] = useState<ChatDocument[]>(() => getAllChats());
  const [processingFiles, setProcessingFiles] = useState<ProcessingFile[]>([]);
  const [showProcessingDialog, setShowProcessingDialog] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [showChunkViewer, setShowChunkViewer] = useState(false);
  const [chunkViewerDocId, setChunkViewerDocId] = useState<string>("");
  const [chunkViewerDocName, setChunkViewerDocName] = useState<string>("");
  const [selectedChunks, setSelectedChunks] = useState<Record<string, number[]>>({});
  const [isSidebarOpen, setIsSidebarOpen] = useState(() => {
    if (typeof window !== "undefined") {
      return window.innerWidth >= 640;
    }
    return true;
  });

  useEffect(() => {
    const allComplete = processingFiles.length > 0 && 
      processingFiles.every(f => f.status === 'done' || f.status === 'failed');
    
    if (allComplete && showProcessingDialog) {
      const timer = setTimeout(() => {
        setShowProcessingDialog(false);
        setProcessingFiles([]);
        setIsProcessing(false);
      }, 3000);
      
      return () => clearTimeout(timer);
    }
  }, [processingFiles, showProcessingDialog]);

  const getModelDisplayName = (modelName?: string) => {
    if (!modelName) return null;
    const modelInfo = LOCAL_MODELS.find(m => m.name === modelName);
    if (modelInfo) {
      return `${modelInfo.name} (${modelInfo.company})`;
    }
    return modelName;
  };

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
      const tempIds = fileArray.map(() => `temp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
      
      // Show processing dialog and block operations
      setIsProcessing(true);
      setShowProcessingDialog(true);
      setProcessingFiles(fileArray.map(file => ({
        name: file.name,
        status: 'uploading',
        progress: 0,
      })));
      
      const placeholders: FileWithUrl[] = fileArray.map((file, idx) => ({
        name: file.name,
        type: file.type,
        size: file.size,
        chatId,
        enabled: true, 
        uploadStatus: 'uploading',
        processingStatus: 'idle',
        downloadStatus: 'idle',
        statusMessage: 'Uploading',
        key: tempIds[idx], // Temporary unique ID
      }));
      setFiles((prev) => [...prev, ...placeholders]);

      const uploaded: FileWithUrl[] = await Promise.all(
        fileArray.map(async (file, idx) => {
          const tempId = tempIds[idx];
          try {
            const { url, key, filename } = await uploadDocument(file, { chatFolder });
            
            // Update processing banner
            setProcessingFiles(prev => prev.map(pf => 
              pf.name === file.name && pf.status === 'uploading'
                ? { ...pf, status: 'processing' as const, progress: 10 }
                : pf
            ));
            
            setFiles((prev) => prev.map(f => (f.key === tempId && f.chatId === chatId ? {
              ...f,
              url,
              key,
              name: filename || f.name,
              uploadStatus: 'uploaded',
              statusMessage: isUsingCustomBackend() ? 'Ready' : 'Uploaded',
              localFile: isUsingCustomBackend() ? file : undefined,
            } : f)));
            return {
              name: filename || file.name,
              url,
              key,
              type: file.type,
              size: file.size,
              chatId,
              enabled: true,
              uploadStatus: 'uploaded',
              statusMessage: isUsingCustomBackend() ? 'Ready' : 'Uploaded',
              localFile: isUsingCustomBackend() ? file : undefined,
            } as FileWithUrl;
          } catch (err) {
            setFiles((prev) => prev.map(f => (f.key === tempId && f.chatId === chatId ? { ...f, uploadStatus: 'failed', statusMessage: 'Upload failed' } : f)));
            return {
              name: file.name,
              type: file.type,
              size: file.size,
              chatId,
              uploadStatus: 'failed',
              statusMessage: 'Upload failed',
            } as FileWithUrl;
          }
        })
      );
      const documentIdByKey = new Map<string, string>();
      for (const uploadedFile of uploaded) {
        if (!uploadedFile.key) {
          setFiles((prev) => prev.map(f => (f.name === uploadedFile.name && f.chatId === chatId ? { ...f, processingStatus: 'failed', statusMessage: 'Missing key' } : f)));
          continue;
        }
        setFiles((prev) => prev.map(f => (f.key === uploadedFile.key && f.chatId === chatId ? { ...f, processingStatus: 'processing', statusMessage: 'Processing' } : f)));
        
        // Update processing banner - show analyzing
        setProcessingFiles(prev => prev.map(pf => 
          pf.name === uploadedFile.name && pf.status === 'processing'
            ? { ...pf, progress: 30 }
            : pf
        ));
        
        try {
          const result = isUsingCustomBackend() && uploadedFile.localFile
            ? await processDocument(undefined, uploadedFile.localFile)
            : await processDocument(uploadedFile.key);
          if (result?.documentId) {
            documentIdByKey.set(uploadedFile.key, result.documentId);
            
            let attempts = 0;
            const maxAttempts = 300; 
            let lastProgress = 30;
            
            while (attempts < maxAttempts) {
              const progressData = await getProgress(result.documentId) as ProgressData | null;
              
              if (progressData) {
                const progress = progressData.progress || lastProgress;
                lastProgress = progress;
                
                setProcessingFiles(prev => prev.map(pf => 
                  pf.name === uploadedFile.name
                    ? { 
                        ...pf, 
                        progress,
                        chunks: progressData.totalChunks,
                        currentChunk: progressData.currentChunk,
                        status: progressData.status === 'complete' ? 'done' as const :
                               progressData.status === 'failed' ? 'failed' as const : 
                               'processing' as const
                      }
                    : pf
                ));
                
                if (progressData.status === 'complete' || progress >= 100) {
                  setProcessingFiles(prev => prev.map(pf => 
                    pf.name === uploadedFile.name
                      ? { ...pf, status: 'done' as const, progress: 100 }
                      : pf
                  ));
                  await clearProgress(result.documentId);
                  break;
                }
                
                if (progressData.status === 'failed') {
                  throw new Error('Processing failed');
                }
              }
              
              await new Promise(resolve => setTimeout(resolve, 100));
              attempts++;
            }
            
            setFiles((prev) => prev.map(f => (f.key === uploadedFile.key && f.chatId === chatId ? { ...f, documentId: result.documentId, processingStatus: 'done', statusMessage: 'Processed' } : f)));
            toast.success(`Document ${uploadedFile.name} processed for RAG`, { duration: 2000 });
          } else {
            setProcessingFiles(prev => prev.map(pf => 
              pf.name === uploadedFile.name
                ? { ...pf, status: 'failed' as const }
                : pf
            ));
            setFiles((prev) => prev.map(f => (f.key === uploadedFile.key && f.chatId === chatId ? { ...f, processingStatus: 'failed', statusMessage: 'Processing failed' } : f)));
            toast.error(`Failed to process ${uploadedFile.name}`);
          }
        } catch (error) {
          setProcessingFiles(prev => prev.map(pf => 
            pf.name === uploadedFile.name
              ? { ...pf, status: 'failed' as const }
              : pf
          ));
          setFiles((prev) => prev.map(f => (f.key === uploadedFile.key && f.chatId === chatId ? { ...f, processingStatus: 'failed', statusMessage: 'Processing error' } : f)));
          toast.error(`Failed to process ${uploadedFile.name}: ${error instanceof Error ? error.message : 'Unknown error'}`);
        }
      }
      try {
        const mapping = await cloneChatFolderToLocal(
          chatId,
          uploaded.map(f => ({ name: f.name, type: f.type, key: f.key }))
        );
        setFiles((prev) => prev.map(f => {
          if (f.chatId !== chatId) return f;
          const uploadedFile = uploaded.find(u => u.key === f.key);
          if (!uploadedFile) return f;
          return {
            ...f,
            localUrl: mapping[uploadedFile.name] || f.localUrl,
            documentId: uploadedFile.key ? documentIdByKey.get(uploadedFile.key) : f.documentId,
            downloadStatus: mapping[uploadedFile.name] ? 'done' : 'failed',
            statusMessage: mapping[uploadedFile.name] ? 'Available' : 'Download failed',
          };
        }));
      } catch {
        setFiles((prev) => prev.map(f => {
          if (f.chatId !== chatId) return f;
          const uploadedFile = uploaded.find(u => u.key === f.key);
          if (!uploadedFile) return f;
          return {
            ...f,
            documentId: uploadedFile.key ? documentIdByKey.get(uploadedFile.key) : f.documentId,
            downloadStatus: 'failed',
            statusMessage: 'Local copy not available',
          };
        }));
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
    const seen = new Map(existing.map(f => [`${f.key || ''}|${f.name}`, f]));
    for (const f of currentFiles) {
      const uniq = `${f.key || ''}|${f.name}`;
      if (seen.has(uniq)) {
        const existingFile = seen.get(uniq)!;
        Object.assign(existingFile, f);
      } else {
        existing.push(f);
        seen.set(uniq, f);
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
          model: model,
        };
        chat.message_objects.push(botMsg);
        updateChat(chat);
        setChats(getAllChats());
        const fileDocIds = Array.from(new Set(
          currentFiles.filter(f => f.enabled !== false).map(f => f.documentId).filter((id): id is string => !!id)
        ));
        const chatLevelDocIds = Array.from(new Set(
          (chat.fileWithUrl || []).filter(f => f.enabled !== false).map(f => f.documentId).filter((id): id is string => !!id)
        ));
        const documentIds = fileDocIds.length > 0 ? fileDocIds : chatLevelDocIds;
        const hasSelectedChunks = documentIds.some(id => selectedChunks[id] && selectedChunks[id].length > 0);
        const specificChunks = hasSelectedChunks ? 
          Object.fromEntries(
            Object.entries(selectedChunks).filter(([docId, chunks]) => 
              documentIds.includes(docId) && chunks.length > 0
            )
          ) : undefined;

        if (stream) {
          let fullResponse = "";
          await sendExternalChatMessage({
            prompt: promptWithFiles,
            model,
            stream: true,
            documentIds,
            specificChunks,
            onStreamChunk: (chunk: string) => {
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
            onStatusChange: (status: any) => {
              const updatedChat = getChatById(chatId!);
              if (!updatedChat) return;
              const msgIdx = updatedChat.message_objects.findIndex(m => m.message_id === botMsgId);
              if (msgIdx !== -1) {
                updatedChat.message_objects[msgIdx].content = `__STATUS__:${status}`;
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
            documentIds,
            specificChunks,
            onStatusChange: (status: any) => {
              // Update bot message with status
              const updatedChat = getChatById(chatId!);
              if (!updatedChat) return;
              const msgIdx = updatedChat.message_objects.findIndex(m => m.message_id === botMsgId);
              if (msgIdx !== -1) {
                updatedChat.message_objects[msgIdx].content = `__STATUS__:${status}`;
                updateChat(updatedChat);
                setChats(getAllChats());
              }
            },
          });
          // Update bot message in chat
          const updatedChat = getChatById(chatId!);
          if (updatedChat) {
            const msgIdx = updatedChat.message_objects.findIndex(m => m.message_id === botMsgId);
            if (msgIdx !== -1) {
              updatedChat.message_objects[msgIdx].content = chatResponse.response;
              updatedChat.message_objects[msgIdx].citations = chatResponse.citations || [];
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
      {/* Processing Dialog */}
      <Dialog open={showProcessingDialog} onOpenChange={() => {}}>
        <DialogContent className="sm:max-w-[600px]" onInteractOutside={(e) => e.preventDefault()}>
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {processingFiles.every(f => f.status === 'done' || f.status === 'failed') ? (
                <>
                  <span className="text-green-600">✓</span>
                  Processing Complete
                </>
              ) : (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary" />
                  Processing {processingFiles.length} {processingFiles.length === 1 ? 'Document' : 'Documents'}
                </>
              )}
            </DialogTitle>
          </DialogHeader>
          <ProcessingBanner 
            files={processingFiles}
            onDismiss={() => {
              setShowProcessingDialog(false);
              setProcessingFiles([]);
              setIsProcessing(false);
            }}
          />
        </DialogContent>
      </Dialog>

      {/* Blocking Overlay */}
      {isProcessing && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-40" />
      )}
      
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
        onPreviewFile={(f) => setPreviewFile(f)}
        onViewChunks={(docId, docName) => {
          setChunkViewerDocId(docId);
          setChunkViewerDocName(docName);
          setShowChunkViewer(true);
        }}
      />

      <main className="flex-1 flex flex-col h-screen overflow-hidden">
        <div className="flex-1 flex flex-col min-h-0 relative">
          <div className="flex-1 flex flex-col">
            <div className="h-[calc(100vh-10rem)] w-full pt-5">
              <ScrollArea className="h-full w-full">
                <div className="space-y-6 p-4 pt-5 max-w-full">
                  {/* ...existing chat message rendering logic... */}
                  {!currentChatId || !getChatById(currentChatId) || getChatById(currentChatId)!.message_objects.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-[calc(100vh-10rem)] p-4">
                      <div className="max-w-2xl w-full space-y-6">
                        <div className="text-center space-y-2">
                          <MessageSquare className="h-12 w-12 text-primary mx-auto mb-2" />
                          <h2 className="text-2xl font-semibold">
                            Document Summarizer
                          </h2>
                          <p className="text-sm text-muted-foreground">
                            Upload documents and get intelligent summaries, analysis, and answers to your questions. Perfect for research assistance and beyond.
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
                                {message.author === "ai" && message.model && (
                                  <div className="flex items-center gap-2 mb-1">
                                    <span className="text-xs px-2 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20 font-medium">
                                      {getModelDisplayName(message.model)}
                                    </span>
                                  </div>
                                )}
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
                                      <PulseLoader />
                                    </span>
                                  ) : message.author === "ai" && message.content.startsWith("__STATUS__:") ? (
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
              onViewChunks={(docId, docName) => {
                setChunkViewerDocId(docId);
                setChunkViewerDocName(docName);
                setShowChunkViewer(true);
              }}
              model={model}
              setModel={setModel}
              disabled={isProcessing}
              selectedChunksInfo={
                Object.keys(selectedChunks).length > 0
                  ? `${Object.values(selectedChunks).reduce((sum, arr) => sum + arr.length, 0)} chunks selected`
                  : undefined
              }
            />
          </div>
        </div>
      </main>

      {previewFile && (
        <FilePreview
          file={previewFile}
          onClose={() => setPreviewFile(null)}
          onViewChunks={(docId, docName) => {
            setChunkViewerDocId(docId);
            setChunkViewerDocName(docName);
            setShowChunkViewer(true);
          }}
        />
      )}

      <ChunkViewer
        isOpen={showChunkViewer}
        onClose={() => setShowChunkViewer(false)}
        documentId={chunkViewerDocId}
        documentName={chunkViewerDocName}
        onApplySelection={(chunkIds) => {
          setSelectedChunks(prev => ({
            ...prev,
            [chunkViewerDocId]: chunkIds
          }));
          toast.success(`Selected ${chunkIds.length} chunks from ${chunkViewerDocName}`, {
            description: 'These chunks will be used in your next query',
            duration: 3000,
          });
        }}
      />

      <Dialog open={showAttachments} onOpenChange={setShowAttachments}>
        <DialogContent className="w-[95vw] sm:w-[85vw] lg:w-[70vw] max-w-none h-[85vh]">
          <DialogHeader>
            <DialogTitle>Attached Files</DialogTitle>
          </DialogHeader>
          <div className="flex-1 overflow-y-auto px-1">
            <FileList
              files={(currentChatId && getChatById(currentChatId)?.fileWithUrl?.length)
                ? (getChatById(currentChatId)!.fileWithUrl)
                : getAllAttachedFiles()}
              onRemove={() => {}}
              previewFile={setPreviewFile}
              onViewChunks={(docId, docName) => {
                console.log('ChatInterface onViewChunks called:', { docId, docName });
                setChunkViewerDocId(docId);
                setChunkViewerDocName(docName);
                setShowChunkViewer(true);
                setShowAttachments(false);
                console.log('State updated, showChunkViewer should be true');
              }}
              onToggleEnabled={(index, enabled) => {
                if (currentChatId) {
                  const chat = getChatById(currentChatId);
                  if (chat && chat.fileWithUrl) {
                    chat.fileWithUrl[index].enabled = enabled;
                    updateChat(chat);
                    setFiles(prev => prev.map((f, i) => 
                      f.chatId === currentChatId && f.key === chat.fileWithUrl[index].key
                        ? { ...f, enabled }
                        : f
                    ));
                  }
                }
              }}
            />
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
