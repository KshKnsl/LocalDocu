"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { LOCAL_MODELS } from "@/lib/localModels";
import { Loader2 } from "lucide-react";

interface NewChatDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreate: (payload: { chatId: string; title: string; description?: string; models: string[] }) => Promise<void>;
}

export default function NewChatDialog({ open, onOpenChange, onCreate }: NewChatDialogProps) {
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [selectedModels, setSelectedModels] = useState<string[]>(["remote"]);
  const [customModel, setCustomModel] = useState("");
  const [isCreating, setIsCreating] = useState(false);

  const toggleModel = (modelName: string) => {
    setSelectedModels((prev) => {
      if (prev.includes(modelName)) return prev.filter(m => m !== modelName);
      return [...prev, modelName];
    });
  };

  const canCreate = title.trim().length > 0 && (selectedModels.length > 0 || customModel.trim().length > 0);

  const handleCreate = async () => {
    if (!canCreate || isCreating) return;
    setIsCreating(true);
    try {
      const chatId = Math.random().toString(36).substring(7);
      const models = [...selectedModels];
      if (customModel.trim()) {
        models.push(customModel.trim());
      }
      await onCreate({ chatId, title: title.trim(), description: description.trim(), models });
      setTitle("");
      setDescription("");
      setSelectedModels(["remote"]);
      setCustomModel("");
      onOpenChange(false);
    } catch (error) {
      console.error("Failed to create chat:", error);
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Create a New Chat</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <div>
            <label className="text-sm text-muted-foreground">Title</label>
            <Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Enter a title" />
          </div>
          <div>
            <label className="text-sm text-muted-foreground">Description</label>
            <Textarea value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Describe the purpose of this chat (optional)" rows={3} />
          </div>
          <div>
            <label className="text-sm text-muted-foreground">Select Models (required)</label>
            <div className="grid grid-cols-2 gap-2 mt-2">
              {LOCAL_MODELS.map((m) => (
                <div key={m.name} className="flex items-center gap-2">
                  <Checkbox checked={selectedModels.includes(m.name)} onCheckedChange={() => toggleModel(m.name)} id={`model-${m.name}`} />
                  <div>
                    <div className="text-sm font-medium">{m.name}</div>
                    <div className="text-xs text-muted-foreground">{m.company} · {m.bestAt}</div>
                  </div>
                </div>
              ))}
            </div>
            <p className="text-xs text-muted-foreground mt-1">You can select multiple models now; these will be locked for this chat and cannot be changed later.</p>
          </div>
          <div>
            <label className="text-sm text-muted-foreground">Custom Model (optional)</label>
            <Input value={customModel} onChange={(e) => setCustomModel(e.target.value)} placeholder="Enter custom model name" />
            <p className="text-xs text-muted-foreground mt-1">Enter a custom model name if not listed above.</p>
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={isCreating}>Cancel</Button>
          <Button onClick={handleCreate} disabled={!canCreate || isCreating}>
            {isCreating ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Creating...
              </>
            ) : (
              "Create"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}