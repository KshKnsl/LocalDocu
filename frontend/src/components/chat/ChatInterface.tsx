"use client";

import React, { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  getAllChats,
  getChatById,
  addChat,
  updateChat,
  ChatDocument,
  MessageObject
} from "@/lib/chatStorage";
import { ScrollArea } from "@/components/ui/scroll-area";
import { FilePreview } from "@/components/FilePreview";
import type { FileWithUrl } from "@/components/ui/FileWithUrl";
import { LOCAL_MODELS } from "@/lib/localModels";
import {
  sendExternalChatMessage,
} from "@/lib/api";
import { loadModel, isUsingCustomBackend } from "@/lib/api";
import { useUser } from "@clerk/nextjs";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ChatSidebar } from "@/components/ChatSidebar";
import NewChatDialog from "@/components/NewChatDialog";
import { ChatInput } from "@/components/ChatInput";
import { FileList } from "@/components/ui/file-list";
import { cloneChatFolderToLocal } from "@/lib/localFiles";
import { ProcessingBanner } from "@/components/ProcessingBanner";
import { ChunkViewer } from "@/components/ChunkViewer";
import { ChatMessage } from "./ChatMessage";
import { ChatEmptyState } from "./ChatEmptyState";
import { ProcessingFile } from "./types";
import { handleFileUpload } from "./fileHandlers";
import { ArrowLeft } from "lucide-react";

interface ChatInterfaceProps {
  initialChatId?: string;
}

export function ChatInterface({ initialChatId }: ChatInterfaceProps) {
  const router = useRouter();
  const [currentChatId, setCurrentChatId] = useState<string | undefined>(initialChatId);
  const [input, setInput] = useState("");
  const [files, setFiles] = useState<FileWithUrl[]>([]);
  const [model, setModel] = useState("mistral");
  const [previewFile, setPreviewFile] = useState<FileWithUrl | string | null>(null);
  const [showAttachments, setShowAttachments] = useState(false);
  const [chats, setChats] = useState<ChatDocument[]>(() => getAllChats());
  const [processingFiles, setProcessingFiles] = useState<ProcessingFile[]>([]);
  const [showProcessingDialog, setShowProcessingDialog] = useState(false);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
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

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { user } = useUser();

  const messages = currentChatId ? getChatById(currentChatId)?.message_objects || [] : [];
  const currentChat = currentChatId ? getChatById(currentChatId) : undefined;
  const allowedModels = currentChat?.models;
  const modelLocked = false;

  useEffect(() => {
    if (initialChatId) {
      setCurrentChatId(initialChatId);
    }
  }, [initialChatId]);

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
    if (!modelName) return undefined;
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

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const fileArray = Array.from(e.target.files);
      let chatId = currentChatId;
      if (!chatId) {
        setCreateDialogOpen(true);
        return;
      }
      const chatFolder = `chats/${chatId}`;
      
      await handleFileUpload(
        fileArray,
        chatId,
        chatFolder,
        setFiles,
        setProcessingFiles,
        setIsProcessing,
        setShowProcessingDialog
      );

      const mapping = await cloneChatFolderToLocal(
        chatId,
        files.map(f => ({ name: f.name, type: f.type, key: f.key }))
      );
      setFiles((prev) => prev.map(f => {
        if (f.chatId !== chatId) return f;
        return {
          ...f,
          localUrl: mapping[f.name] || f.localUrl,
          downloadStatus: mapping[f.name] ? 'done' : 'failed',
          statusMessage: mapping[f.name] ? 'Available' : 'Download failed',
        };
      }));
    }
  };

  const handleRemoveFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSidebarChatSelect = (chatId?: string) => {
    if (chatId) {
      router.push(`/chat/${chatId}`);
    }
  };

  const handleSidebarNewChat = () => {
    setCreateDialogOpen(true);
  };

  const handleSubmit = async () => {
    if (!input.trim() && files.length === 0) return;
    let chatId = currentChatId;
    let chat = chatId ? getChatById(chatId) : undefined;
    if (!chat) {
      setCreateDialogOpen(true);
      return;
    }
    const messageId = Math.random().toString(36).substring(7);
    const now = new Date().toISOString();
    const currentInput = input.trim();
    const currentFiles = [...files];

    const promptWithFiles = currentInput;

    const userMsg: MessageObject = {
      message_id: messageId,
      author: "user",
      content: currentInput,
      created_at: now,
      files: currentFiles,
    };
    chat.message_objects.push(userMsg);
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

        const chatResponse = await sendExternalChatMessage({
          prompt: promptWithFiles,
          model,
          documentIds,
          specificChunks,
        });
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
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Something went wrong");
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
        onPreviewFile={(f) => setPreviewFile(f)}
      />
      <NewChatDialog
        open={createDialogOpen}
        onOpenChange={setCreateDialogOpen}
        onCreate={async ({ chatId, title, description, models }) => {
          if (isUsingCustomBackend()) {
            toast.info("Pulling selected models...");
            try {
              for (const model of models) {
                await loadModel(model);
              }
              toast.success("All models pulled successfully!");
            } catch (error) {
              toast.error("Failed to pull some models. Please try again.");
              throw error;
            }
          }
          const now = new Date().toISOString();
          addChat({ chat_id: chatId, title, created_at: now, fileWithUrl: [], message_objects: [], description, models });
          setChats(getAllChats());
          router.push(`/chat/${chatId}`);
        }}
      />

      <main className="flex-1 flex flex-col h-screen overflow-hidden">
        <div className="border-b bg-background p-3">
          <div className="flex items-center gap-3">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => router.push('/spaces')}
              className="gap-2"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Spaces
            </Button>
            <div className="flex-1" />
          </div>
        </div>

        <div className="flex-1 flex flex-col min-h-0 relative">
          <div className="flex-1 flex flex-col">
            <div className="h-[calc(100vh-10rem)] w-full pt-5">
              <ScrollArea className="h-full w-full">
                <div className="space-y-6 p-4 pt-5 max-w-full">
                  {!currentChatId || !getChatById(currentChatId) || getChatById(currentChatId)!.message_objects.length === 0 ? (
                    <ChatEmptyState />
                  ) : (
                    messages.map((message, idx, arr) => {
                      const isLast = idx === arr.length - 1;
                      return (
                        <ChatMessage
                          key={message.message_id}
                          message={message}
                          isLast={isLast}
                          userImageUrl={user?.imageUrl}
                          userName={user?.fullName || user?.username || undefined}
                          modelDisplayName={message.model ? getModelDisplayName(message.model) : undefined}
                          onPreviewFile={setPreviewFile}
                        />
                      );
                    })
                  )}
                  <div ref={messagesEndRef} />
                </div>
              </ScrollArea>
            </div>
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
              modelLocked={modelLocked}
              allowedModels={allowedModels}
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
              onViewChunks={(docId, docName) => {
                setChunkViewerDocId(docId);
                setChunkViewerDocName(docName);
                setShowChunkViewer(true);
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

export default ChatInterface;
