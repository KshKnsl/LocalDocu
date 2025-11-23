"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { getAllChats, deleteChat, ChatDocument, addChat } from "@/lib/chatStorage";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Plus, Search, MessageSquare, Trash2, FileText, Calendar, Download } from "lucide-react";
import { toast } from "sonner";
import { UserButton } from "@clerk/nextjs";
import ThemeSwitcher from "@/components/ui/theme-switcher";
import { exportChatToPDF, exportAllChatsToPDF } from "@/lib/pdfExport";
import { getChatPreview, getMessageCount, getFileCount, filterChatsByQuery } from "@/lib/chatUtils";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import NewChatDialog from "@/components/NewChatDialog";
import { BackendConfigDialog } from "@/components/BackendConfig";
import { loadModel, isUsingCustomBackend } from "@/lib/api";

export default function SpacesPage() {
  const router = useRouter();
  const [chats, setChats] = useState<ChatDocument[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [chatToDelete, setChatToDelete] = useState<string | null>(null);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);

  useEffect(() => setChats(getAllChats()), []);

  const handleCreateSpace = () => setCreateDialogOpen(true);
  const handleOpenSpace = (chatId: string) => router.push(`/chat/${chatId}`);
  const handleDeleteClick = (chatId: string, e: React.MouseEvent) => { e.stopPropagation(); setChatToDelete(chatId); setDeleteDialogOpen(true); };
  const handleDeleteConfirm = () => { if (chatToDelete) { deleteChat(chatToDelete); setChats(getAllChats()); toast.success("Document space deleted"); setDeleteDialogOpen(false); setChatToDelete(null); } };
  const handleExportChat = (chat: ChatDocument, e: React.MouseEvent) => { e.stopPropagation(); exportChatToPDF(chat); toast.success("Chat exported as PDF"); };
  const handleExportAll = () => { if (!chats.length) return toast.error("No chats to export"); exportAllChatsToPDF(chats); toast.success(`Exported ${chats.length} chats as PDF`); };

  const filteredChats = filterChatsByQuery(chats, searchQuery);

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20 w-screen">
      <header className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <img src="/logo.png" alt="App Logo" className="w-10 h-10 rounded-lg" />
              <div><h1 className="text-2xl font-bold">Document Spaces</h1><p className="text-sm text-muted-foreground">Manage your research conversations</p></div>
            </div>
            <div className="flex items-center gap-2 sm:gap-3 w-full sm:w-auto justify-end">
              <Button variant="outline" size="sm" onClick={handleExportAll} disabled={!chats.length} className="flex-shrink-0">
                <Download className="h-4 w-4 sm:mr-2" />
                <span className="hidden sm:inline">Export All</span>
              </Button>
              <BackendConfigDialog />
              <ThemeSwitcher />
              <UserButton afterSwitchSessionUrl="/spaces" />
            </div>
          </div>
        </div>
      </header>
      <main className="container mx-auto px-4 py-8">
        <div className="flex gap-4 mb-8">
          <div className="relative flex-1"><Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input placeholder="Search spaces..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} className="pl-10" />
          </div>
          <Button onClick={handleCreateSpace} size="lg" className="gap-2"><Plus className="h-5 w-5" />New Space</Button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          <Card className="p-6"><div className="flex items-center gap-4"><div className="flex items-center justify-center w-12 h-12 rounded-lg bg-primary/10"><MessageSquare className="h-6 w-6 text-primary" /></div><div><p className="text-2xl font-bold">{chats.length}</p><p className="text-sm text-muted-foreground">Total Spaces</p></div></div></Card>
          <Card className="p-6"><div className="flex items-center gap-4"><div className="flex items-center justify-center w-12 h-12 rounded-lg bg-blue-500/10"><FileText className="h-6 w-6 text-blue-500" /></div><div><p className="text-2xl font-bold">{chats.reduce((s, c) => s + getFileCount(c), 0)}</p><p className="text-sm text-muted-foreground">Documents</p></div></div></Card>
          <Card className="p-6"><div className="flex items-center gap-4"><div className="flex items-center justify-center w-12 h-12 rounded-lg bg-green-500/10"><MessageSquare className="h-6 w-6 text-green-500" /></div><div><p className="text-2xl font-bold">{chats.reduce((s, c) => s + getMessageCount(c), 0)}</p><p className="text-sm text-muted-foreground">Messages</p></div></div></Card>
        </div>
        {!filteredChats.length ? (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="flex items-center justify-center w-20 h-20 rounded-full bg-muted mb-4"><MessageSquare className="h-10 w-10 text-muted-foreground" /></div>
            <h3 className="text-xl font-semibold mb-2">{searchQuery ? "No spaces found" : "No document spaces yet"}</h3>
            <p className="text-muted-foreground mb-6 text-center max-w-md">{searchQuery ? "Try adjusting your search query" : "Create your first document space to start researching with AI"}</p>
            {!searchQuery && <Button onClick={handleCreateSpace} size="lg" className="gap-2"><Plus className="h-5 w-5" />Create Your First Space</Button>}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredChats.map((chat) => (
              <Card key={chat.chat_id} className="group cursor-pointer hover:shadow-lg transition-all duration-200 hover:scale-[1.02] touch-manipulation select-none" onClick={() => handleOpenSpace(chat.chat_id)}>
                <div className="p-6">
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1 min-w-0"><h3 className="text-lg font-semibold mb-1 truncate group-hover:text-primary transition-colors">{chat.title}</h3>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground"><Calendar className="h-3 w-3" />{new Date(chat.created_at).toLocaleDateString()}</div>
                    </div>
                    <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <Button variant="ghost" size="icon" className="h-8 w-8" onClick={(e) => handleExportChat(chat, e)} title="Export as PDF"><Download className="h-4 w-4" /></Button>
                      <Button variant="ghost" size="icon" className="h-8 w-8 text-destructive hover:text-destructive hover:bg-destructive/10" onClick={(e) => handleDeleteClick(chat.chat_id, e)} title="Delete space"><Trash2 className="h-4 w-4" /></Button>
                    </div>
                  </div>
                  <p className="text-sm text-muted-foreground mb-4 line-clamp-2 min-h-[2.5rem]">{getChatPreview(chat)}</p>
                  {chat.description && (
                    <p className="text-sm text-muted-foreground mb-4 italic">{chat.description}</p>
                  )}
                  <div className="mb-4">
                    <div className="text-xs text-muted-foreground mb-1">Models:</div>
                    <div className="flex flex-wrap gap-1">
                      {chat.models && chat.models.length > 0 ? (
                        chat.models.map(m => (
                          <span key={m} className="text-xs bg-primary/10 text-primary px-2 py-1 rounded">{m}</span>
                        ))
                      ) : (
                        <span className="text-xs text-muted-foreground">None</span>
                      )}
                    </div>
                  </div>
                  <div className="mb-4">
                    <div className="text-xs text-muted-foreground mb-1">Files:</div>
                    <div className="text-xs text-muted-foreground">
                      {chat.fileWithUrl.length > 0 ? (
                        chat.fileWithUrl.slice(0, 3).map(f => f.name).join(', ') + (chat.fileWithUrl.length > 3 ? '...' : '')
                      ) : (
                        'No files'
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-4 text-xs text-muted-foreground pt-4 border-t">
                    <div className="flex items-center gap-1"><MessageSquare className="h-3 w-3" /><span>{getMessageCount(chat)} messages</span></div>
                    <div className="flex items-center gap-1"><FileText className="h-3 w-3" /><span>{getFileCount(chat)} files</span></div>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
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
      </main>
      <Dialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <DialogContent className="w-[95vw] max-w-none sm:max-w-md"><DialogHeader><DialogTitle>Delete Document Space?</DialogTitle><DialogDescription>This action cannot be undone. This will permanently delete the document space and all its messages and files.</DialogDescription></DialogHeader>
          <DialogFooter><Button variant="outline" onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
            <Button variant="outline" className="text-destructive hover:text-destructive hover:bg-destructive/10" onClick={handleDeleteConfirm}>Delete</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
