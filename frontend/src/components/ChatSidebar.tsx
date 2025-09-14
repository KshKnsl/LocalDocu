"use client";

import React, { useState } from "react";
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
} from "lucide-react";
import ThemeSwitcher from "./Theme-switcher";

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
  stream?: boolean;
  setStream?: (v: boolean) => void;
}

export function ChatSidebar({
  isSidebarOpen,
  setIsSidebarOpen,
  initialChats = [],
  onChatSelect,
  onNewChatStart,
  stream = false,
  setStream,
}: ChatSidebarProps) {
  const [chats, setChats] = useState<Chat[]>(initialChats);
  // Removed selectedChatId and editingChatId for simplicity

  const handleNewChat = () => {
    const newChat: Chat = {
      id: Math.random().toString(36).substring(7),
      title: "New Chat",
      timestamp: new Date(),
      preview: "Start a new conversation...",
    };
    setChats((prev) => [newChat, ...prev]);
    onNewChatStart?.();
  };

  const handleSelectChat = (chatId: string) => {
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
            <ChatSidebarItem
              key={chat.id}
              chat={chat}
              isSidebarOpen={isSidebarOpen}
              onSelect={handleSelectChat}
              onSave={handleRenameChat}
              onDelete={handleDeleteChat}
            />
          ))}
        </div>
      </ScrollArea>

      <div className="border-t bg-muted/50 p-4">
        {isSidebarOpen ? (
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Switch
                checked={stream}
                onCheckedChange={setStream}
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
        ) : (
          <div className="flex flex-col items-center gap-2">
            <Switch
              checked={stream}
              onCheckedChange={setStream}
              id="stream-toggle"
              className="mb-2"
              title="Stream"
            />
            <UserButton afterSwitchSessionUrl="/" />
            <ThemeSwitcher />
          </div>
        )}
      </div>

    </aside>
  );
}
