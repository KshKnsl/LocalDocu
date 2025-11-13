import React, { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import { MessageSquare, Trash2, Edit2, Check, X, FileDown } from "lucide-react";
import { ChatDocument } from "@/lib/chatStorage";
import { Input } from "@/components/ui/input";
import { exportChatToPDF } from "@/lib/pdfExport";
import { toast } from "sonner";

interface ChatSidebarItemProps {
  chat: ChatDocument;
  isSidebarOpen: boolean;
  isSelected?: boolean;
  onSelect: (id: string) => void;
  onSave: (id: string, newTitle: string) => void;
  onDelete: (id: string) => void;
}

const IconBtn = ({ icon: Icon, onClick, color, title }: { icon: any; onClick: () => void; color?: string; title?: string }) => (
  <div role="button" tabIndex={0} title={title} className={cn("inline-flex items-center justify-center rounded-md h-6 w-6 text-sm font-medium ring-offset-background transition-colors hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2", color)} onClick={(e) => { e.stopPropagation(); onClick(); }} onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.stopPropagation(); onClick(); } }}>
    <Icon className="h-3 w-3" />
  </div>
);

export function ChatSidebarItem({ chat, isSidebarOpen, isSelected, onSelect, onSave, onDelete }: ChatSidebarItemProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editTitle, setEditTitle] = useState(chat.title);
  useEffect(() => setEditTitle(chat.title), [chat.title]);

  const handleSave = () => { if (editTitle.trim() && editTitle !== chat.title) onSave(chat.chat_id, editTitle.trim()); setIsEditing(false); };
  const handleCancel = () => { setEditTitle(chat.title); setIsEditing(false); };
  const handleExport = () => { exportChatToPDF(chat); toast.success("Chat exported as PDF"); };

  return (
    <div onClick={() => !isEditing && onSelect(chat.chat_id)} className={cn("w-full flex flex-col gap-1 rounded-lg border px-3 py-2 text-sm transition-colors hover:bg-accent text-left cursor-pointer", isSelected && "bg-accent border-primary", isEditing && "cursor-default")} role="button" tabIndex={0} onKeyDown={(e) => !isEditing && (e.key === "Enter" || e.key === " ") && onSelect(chat.chat_id)}>
      {isSidebarOpen ? (
        <>
          <div className="flex items-center gap-2 group">
            <MessageSquare className="h-4 w-4 shrink-0" />
            <div className="flex-1 min-w-0">
              {isEditing ? <Input value={editTitle} onChange={(e) => setEditTitle(e.target.value)} onKeyDown={(e) => e.key === "Enter" ? handleSave() : e.key === "Escape" && handleCancel()} onBlur={handleSave} className="h-6 text-sm font-medium border-none p-0 focus-visible:ring-0" autoFocus onClick={(e) => e.stopPropagation()} /> : <span className="font-medium line-clamp-1 cursor-pointer hover:text-primary">{chat.title}</span>}
            </div>
            <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              {isEditing ? (<><IconBtn icon={Check} onClick={handleSave} color="hover:text-green-600" /><IconBtn icon={X} onClick={handleCancel} color="hover:text-destructive" /></>) : (<><IconBtn icon={FileDown} onClick={handleExport} color="hover:text-blue-600" title="Export as PDF" /><IconBtn icon={Edit2} onClick={() => setIsEditing(true)} color="hover:text-primary" /><IconBtn icon={Trash2} onClick={() => onDelete(chat.chat_id)} color="hover:text-destructive" /></>)}
            </div>
          </div>
          <span className="text-xs text-muted-foreground line-clamp-1">{chat.message_objects.length > 0 ? chat.message_objects[0].content.slice(0, 40) : "Start a new conversation..."}</span>
          <span className="text-xs text-muted-foreground mt-1">{new Date(chat.created_at).toLocaleDateString()}</span>
        </>
      ) : <div className="flex justify-center"><MessageSquare className="h-4 w-4" /></div>}
    </div>
  );
}
