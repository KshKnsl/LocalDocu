"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import { UserButton } from "@clerk/nextjs";
import {
  MessageSquare,
  Plus,
  ChevronLeft,
  ChevronRight,
  Pencil,
  Trash2,
} from "lucide-react";
import ThemeSwitcher from "./Theme-switcher";
import { InputLabel } from "./ui/input-label";


interface Chat {
  id: string;
  title: string;
  timestamp: Date;
  preview: string;
}

interface ChatSidebarProps {
  isSidebarOpen: boolean;
  setIsSidebarOpen: (open: boolean) => void;
  initialChats?: Chat[];
  onChatSelect?: (chatId: string) => void;
  onNewChatStart?: () => void;
}

export function ChatSidebar({
  isSidebarOpen,
  setIsSidebarOpen,
  initialChats = [],
  onChatSelect,
  onNewChatStart,
}: ChatSidebarProps) {
  const [chats, setChats] = useState<Chat[]>(initialChats);
  const [selectedChatId, setSelectedChatId] = useState<string | undefined>();
  const [editingChatId, setEditingChatId] = useState<string | null>(null);

  const handleNewChat = () => {
    const newChat: Chat = {
      id: Math.random().toString(36).substring(7),
      title: "New Chat",
      timestamp: new Date(),
      preview: "Start a new conversation...",
    };
    setChats((prev) => [newChat, ...prev]);
    setSelectedChatId(newChat.id);
    onNewChatStart?.();
  };

  const handleSelectChat = (chatId: string) => {
    setSelectedChatId(chatId);
    onChatSelect?.(chatId);
  };

  const handleRenameChat = (chatId: string, newTitle: string) => {
    setChats((prevChats) =>
      prevChats.map((chat) =>
        chat.id === chatId ? { ...chat, title: newTitle } : chat
      )
    );
  };

  const handleDeleteChat = (chatId: string) => {
    setChats((prevChats) => prevChats.filter((chat) => chat.id !== chatId));
    if (selectedChatId === chatId) {
      handleNewChat();
    }
  };
  return (
    <aside
      className={cn(
        "relative border-r bg-muted/50 flex flex-col transition-all duration-300 ease-in-out h-screen",
        isSidebarOpen ? "w-[300px]" : "w-[50px]"
      )}
    >
      <div className="p-2 border-b relative">
        {isSidebarOpen ? (
          <Button
            onClick={handleNewChat}
            className="w-full justify-start gap-2"
          >
            <Plus className="h-4 w-4" />
            New Chat
          </Button>
        ) : (
          <Button size="icon" onClick={handleNewChat} className="w-full">
            <Plus className="h-4 w-4" />
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
      </div>

      <ScrollArea className="flex-1 overflow-y-auto">
        <div className="p-2 space-y-2">
          {chats.map((chat) => (
            <div
              key={chat.id}
              onClick={() => handleSelectChat(chat.id)}
              className={cn(
                "w-full flex flex-col gap-1 rounded-lg border px-3 py-2 text-sm transition-colors hover:bg-accent text-left cursor-pointer",
                selectedChatId === chat.id ? "bg-accent" : "transparent"
              )}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  handleSelectChat(chat.id);
                }
              }}
            >
              {isSidebarOpen ? (
                <>
                  <div className="flex items-center gap-2 group">
                    <MessageSquare className="h-4 w-4 shrink-0" />
                    <div className="flex-1 min-w-0">
                      <InputLabel
                        value={chat.title}
                        isEditing={editingChatId === chat.id}
                        onEdit={() => setEditingChatId(chat.id)}
                        onSave={(newTitle) => {
                          handleRenameChat(chat.id, newTitle);
                          setEditingChatId(null);
                        }}
                        onCancel={() => setEditingChatId(null)}
                        onChange={() => {}}
                        className="font-medium line-clamp-1 cursor-pointer hover:text-primary"
                      />
                    </div>
                    {editingChatId !== chat.id && (
                      <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                        <div
                          role="button"
                          tabIndex={0}
                          className="inline-flex items-center justify-center rounded-md h-6 w-6 text-sm font-medium ring-offset-background transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                          onClick={(e) => {
                            e.stopPropagation();
                            setEditingChatId(chat.id);
                          }}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.stopPropagation();
                              setEditingChatId(chat.id);
                            }
                          }}
                        >
                          <Pencil className="h-3 w-3" />
                        </div>
                        <div
                          role="button"
                          tabIndex={0}
                          className="inline-flex items-center justify-center rounded-md h-6 w-6 text-sm font-medium ring-offset-background transition-colors hover:bg-accent hover:text-destructive focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleDeleteChat(chat.id);
                          }}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.stopPropagation();
                              handleDeleteChat(chat.id);
                            }
                          }}
                        >
                          <Trash2 className="h-3 w-3" />
                        </div>
                      </div>
                    )}
                  </div>
                  <span className="text-xs text-muted-foreground line-clamp-1">
                    {chat.preview}
                  </span>
                  <span className="text-xs text-muted-foreground mt-1">
                    {chat.timestamp.toLocaleDateString()}
                  </span>
                </>
              ) : (
                <div className="flex justify-center">
                  <MessageSquare className="h-4 w-4" />
                </div>
              )}
            </div>
          ))}
        </div>
      </ScrollArea>

      <div className="border-t bg-muted/50 p-4">
        {isSidebarOpen ? (
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 min-w-0 group">
              <div className="flex items-center gap-1 min-w-0">
                {selectedChatId && editingChatId === selectedChatId ? (
                  <InputLabel
                    value={chats.find((chat) => chat.id === selectedChatId)?.title || ""}
                    isEditing={true}
                    onEdit={() => {}}
                    onSave={(newTitle) => {
                      handleRenameChat(selectedChatId, newTitle);
                      setEditingChatId(null);
                    }}
                    onCancel={() => setEditingChatId(null)}
                    onChange={() => {}}
                    className="font-semibold"
                  />
                ) : (
                  <h2 className="font-semibold truncate">
                    {selectedChatId
                      ? chats.find((chat) => chat.id === selectedChatId)?.title
                      : "New Chat"}
                  </h2>
                )}
                {selectedChatId && editingChatId !== selectedChatId && (
                  <div
                    role="button"
                    tabIndex={0}
                    className="inline-flex items-center justify-center rounded-md h-6 w-6 text-sm font-medium ring-offset-background transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 opacity-0 group-hover:opacity-100"
                    onClick={() => setEditingChatId(selectedChatId)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        setEditingChatId(selectedChatId);
                      }
                    }}
                  >
                    <Pencil className="h-3 w-3" />
                  </div>
                )}
              </div>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <ThemeSwitcher />
              <UserButton afterSwitchSessionUrl="/" />
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <UserButton afterSwitchSessionUrl="/" />
            <ThemeSwitcher />
          </div>
        )}
      </div>

    </aside>
  );
}
