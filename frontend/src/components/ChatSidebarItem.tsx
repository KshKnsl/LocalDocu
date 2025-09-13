import React from "react";
import { cn } from "@/lib/utils";
import { MessageSquare, Pencil, Trash2 } from "lucide-react";
import { InputLabel } from "./ui/input-label";

interface ChatSidebarItemProps {
  chat: {
    id: string;
    title: string;
    timestamp: Date;
    preview: string;
  };
  isSidebarOpen: boolean;
  onSelect: (id: string) => void;
  onSave: (id: string, newTitle: string) => void;
  onDelete: (id: string) => void;
}

export function ChatSidebarItem({
  chat,
  isSidebarOpen,
  onSelect,
  onSave,
  onDelete,
}: ChatSidebarItemProps) {
  return (
    <div
      key={chat.id}
      onClick={() => onSelect(chat.id)}
      className={cn(
        "w-full flex flex-col gap-1 rounded-lg border px-3 py-2 text-sm transition-colors hover:bg-accent text-left cursor-pointer"
      )}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          onSelect(chat.id);
        }
      }}
    >
      {isSidebarOpen ? (
        <>
          <div className="flex items-center gap-2 group">
            <MessageSquare className="h-4 w-4 shrink-0" />
            <div className="flex-1 min-w-0">
              <span className="font-medium line-clamp-1 cursor-pointer hover:text-primary">
                {chat.title}
              </span>
            </div>
            <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              <div
                role="button"
                tabIndex={0}
                className="inline-flex items-center justify-center rounded-md h-6 w-6 text-sm font-medium ring-offset-background transition-colors hover:bg-accent hover:text-destructive focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(chat.id);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.stopPropagation();
                    onDelete(chat.id);
                  }
                }}
              >
                <Trash2 className="h-3 w-3" />
              </div>
            </div>
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
  );
}
