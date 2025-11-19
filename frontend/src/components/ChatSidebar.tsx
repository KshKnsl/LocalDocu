"use client";

import React, { useMemo, useState } from "react";
import {
  getAllChats,
  updateChat,
  deleteChat,
  ChatDocument
} from "@/lib/chatStorage";
import { toast } from "sonner";
import { Switch } from "@/components/ui/switch";
import { ChatSidebarItem } from "./ChatSidebarItem";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { UserButton } from "@clerk/nextjs";
import {
  Plus,
  ChevronLeft,
  ChevronRight,
  Eye,
  Trash2,
  Download,
  FileText,
} from "lucide-react";
import ThemeSwitcher from "./ui/theme-switcher";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import type { FileWithUrl } from "@/components/ui/FileWithUrl";
import { FileList } from "@/components/ui/file-list";
import { BackendConfigDialog } from "./BackendConfig";
import { exportChatToPDF, exportAllChatsToPDF } from "@/lib/pdfExport";


interface ChatSidebarProps {
  isSidebarOpen: boolean;
  setIsSidebarOpen: (open: boolean) => void;
  chats: ChatDocument[];
  currentChatId?: string;
  onChatSelect?: (chatId?: string) => void;
  onNewChatStart?: () => void;
  onChatsUpdate?: () => void;
  stream?: boolean;
  setStream?: (v: boolean) => void;
  onPreviewFile?: (file: FileWithUrl) => void;
  onViewChunks?: (documentId: string, documentName: string) => void;
}


export function ChatSidebar({
  isSidebarOpen,
  setIsSidebarOpen,
  chats,
  currentChatId,
  onChatSelect,
  onNewChatStart,
  onChatsUpdate,
  stream = false,
  setStream,
  onPreviewFile,
  onViewChunks,
}: ChatSidebarProps) {
  const [search, setSearch] = useState("");
  const [filesOpen, setFilesOpen] = useState(false);

  const handleSelectChat = (chatId: string) => {
    onChatSelect?.(chatId);
  };

  const handleRenameChat = (chatId: string, newTitle: string) => {
    const chat = getAllChats().find((c) => c.chat_id === chatId);
    if (chat) {
      chat.title = newTitle;
      updateChat(chat);
      onChatsUpdate?.();
    }
  };

  const handleDeleteChat = (chatId: string) => {
    deleteChat(chatId);
    onChatSelect?.(undefined);
    onChatsUpdate?.();
  };
  const isVercel = typeof window !== "undefined" && window.location.hostname.endsWith("vercel.app");

  // Handler for stream toggle
  const handleStreamToggle = (v: boolean) => {
    if (v && isVercel) {
      toast.error("Vercel does not support streaming. Stream has been turned off.");
      setStream?.(false);
    } else {
      setStream?.(v);
    }
  };

  const filteredChats = chats.filter(chat => {
    const term = search.toLowerCase();
    return (
      chat.title.toLowerCase().includes(term) ||
      chat.message_objects.some(m => m.content.toLowerCase().includes(term))
    );
  });

  const allFiles = useMemo(() => {
    const map = new Map<string, FileWithUrl>();
    for (const chat of chats) {
      for (const f of (chat.fileWithUrl || [])) {
        const key = `${f.key || ''}|${f.name}`;
        if (!map.has(key)) map.set(key, { ...f, chatId: f.chatId || chat.chat_id });
      }
      for (const m of (chat.message_objects || [])) {
        for (const f of (m.files || [])) {
          const key = `${f.key || ''}|${f.name}`;
          if (!map.has(key)) map.set(key, { ...f, chatId: f.chatId || chat.chat_id });
        }
      }
    }
    return Array.from(map.values());
  }, [chats]);

  return (
    <aside
      className={cn(
        "relative border-r bg-muted/50 flex flex-col transition-all duration-300 ease-in-out h-screen",
        isSidebarOpen ? "w-[300px]" : "w-[50px]"
      )}
    >
      <div className="p-2 border-b relative">
        {isSidebarOpen && (
          <div className="flex items-center gap-2 mb-2 px-1">
            <img src="/logo.png" alt="Docu" className="w-8 h-8" />
            <span className="text-lg font-bold">Docu</span>
          </div>
        )}
        {isSidebarOpen ? (
          <Button
            onClick={onNewChatStart}
            className="w-full justify-start gap-2"
          >
            <Plus className="h-4 w-4" />
            New Chat
          </Button>
        ) : (
          <Button size="icon" onClick={onNewChatStart} className="w-full" title="LocalDocu">
            <img src="/logo.png" alt="LocalDocu" className="w-6 h-6" />
          </Button>
        )}
        <div className="absolute -right-[18px] top-2 z-50">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="h-8 w-8 rounded-full bg-background shadow-md border hover:bg-accent focus-visible:ring-offset-1"
            title={isSidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
          >
            {isSidebarOpen ? (
              <ChevronLeft className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </Button>
        </div>
        {isSidebarOpen && (
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search chats..."
            className="mt-2 w-full px-2 py-1 rounded border bg-background text-sm focus:outline-none focus:ring"
          />
        )}
        {isSidebarOpen && (
          <div className="mt-2 space-y-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setFilesOpen(true)}
              title="View all files"
              className="w-full justify-start gap-2"
            >
              <Eye className="h-4 w-4" />
              All files
            </Button>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  if (currentChatId) {
                    const chat = chats.find(c => c.chat_id === currentChatId);
                    if (chat) {
                      exportChatToPDF(chat);
                      toast.success("Chat exported as PDF");
                    }
                  } else {
                    toast.error("No chat selected");
                  }
                }}
                title="Export current chat as PDF"
                className="flex-1 justify-start gap-2"
                disabled={!currentChatId}
              >
                <FileText className="h-4 w-4" />
                Export Chat
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  if (chats.length === 0) {
                    toast.error("No chats to export");
                    return;
                  }
                  exportAllChatsToPDF(chats);
                  toast.success(`Exported ${chats.length} chats as PDF`);
                }}
                title="Export all chats as PDF"
                className="flex-1 justify-start gap-2"
              >
                <Download className="h-4 w-4" />
                Export All
              </Button>
            </div>
          </div>
        )}
      </div>

      <ScrollArea className="flex-1 overflow-y-auto">
        <div className="p-2 space-y-2">
          {filteredChats.map((chat) => (
            <ChatSidebarItem
              key={chat.chat_id}
              chat={chat}
              isSidebarOpen={isSidebarOpen}
              isSelected={chat.chat_id === currentChatId}
              onSelect={handleSelectChat}
              onSave={handleRenameChat}
              onDelete={handleDeleteChat}
            />
          ))}
        </div>
      </ScrollArea>

      <div className="border-t bg-muted/50 p-4">
        {isSidebarOpen ? (
          <div className="space-y-3">
            <BackendConfigDialog collapsed={false} />
             <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <Switch
                  checked={stream}
                  onCheckedChange={handleStreamToggle}
                  id="stream-toggle"
                  className="mr-2"
                />
                <label htmlFor="stream-toggle" className="text-xs text-muted-foreground select-none">Stream</label>
              </div>
              <div className="flex items-center gap-2">
                <ThemeSwitcher />
                <UserButton afterSwitchSessionUrl="/" />
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <div className="w-full max-w-[200px]">
              <BackendConfigDialog collapsed={true} />
            </div>
            <Switch
              checked={stream}
              onCheckedChange={handleStreamToggle}
              id="stream-toggle"
              className="mb-2"
              title="Stream"
            />
            <UserButton afterSwitchSessionUrl="/" />
            <ThemeSwitcher />
          </div>
        )}
      </div>

      <Dialog open={filesOpen} onOpenChange={setFilesOpen}>
        <DialogContent className="w-[95vw] sm:w-[85vw] lg:w-[70vw] max-w-none h-[85vh]">
          <DialogHeader>
            <DialogTitle>All Files</DialogTitle>
          </DialogHeader>
          <div className="flex-1 overflow-y-auto px-1">
            <FileList
              files={allFiles}
              onRemove={(index) => {
                const file = allFiles[index];
                if (file.chatId) {
                  const chat = chats.find(c => c.chat_id === file.chatId);
                  if (chat) {
                    chat.fileWithUrl = (chat.fileWithUrl || []).filter(f => 
                      !((file.key && f.key === file.key) || (!file.key && f.name === file.name))
                    );
                    chat.message_objects = chat.message_objects.map(m => ({
                      ...m,
                      files: (m.files || []).filter(f => 
                        !((file.key && f.key === file.key) || (!file.key && f.name === file.name))
                      )
                    }));
                    updateChat(chat);
                    onChatsUpdate?.();
                    toast.success("File deleted");
                  }
                }
              }}
              previewFile={(file) => {
                setFilesOpen(false);
                onPreviewFile?.(file);
              }}
              onViewChunks={onViewChunks}
            />
          </div>
        </DialogContent>
      </Dialog>

    </aside>
  );
}
