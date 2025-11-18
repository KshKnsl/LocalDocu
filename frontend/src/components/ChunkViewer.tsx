'use client';

import React, { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { Search, Check, X } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { getDocumentChunks, type DocumentChunk } from '@/lib/api';
import { toast } from 'sonner';

interface ChunkViewerProps {
  isOpen: boolean;
  onClose: () => void;
  documentId: string;
  documentName: string;
  onApplySelection: (selectedChunkIds: number[]) => void;
}

export function ChunkViewer({ isOpen, onClose, documentId, documentName, onApplySelection }: ChunkViewerProps) {
  const [chunks, setChunks] = useState<DocumentChunk[]>([]);
  const [selectedChunks, setSelectedChunks] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filteredChunks, setFilteredChunks] = useState<DocumentChunk[]>([]);
  const [expandedChunks, setExpandedChunks] = useState<Set<number>>(new Set());

  useEffect(() => {
    if (isOpen && documentId) {
      loadChunks();
    }
  }, [isOpen, documentId]);

  useEffect(() => {
    if (searchQuery.trim() === '') {
      setFilteredChunks(chunks);
    } else {
      const query = searchQuery.toLowerCase();
      setFilteredChunks(
        chunks.filter(chunk => 
          chunk.content.toLowerCase().includes(query) ||
          chunk.metadata.title?.toLowerCase().includes(query)
        )
      );
    }
  }, [searchQuery, chunks]);

  const loadChunks = async () => {
    setLoading(true);
    try {
      const loadedChunks = await getDocumentChunks(documentId);
      setChunks(loadedChunks);
      setFilteredChunks(loadedChunks);
      toast.success(`Loaded ${loadedChunks.length} chunks from document`);
    } catch (error) {
      toast.error('Failed to load chunks', {
        description: error instanceof Error ? error.message : 'Unknown error'
      });
    } finally {
      setLoading(false);
    }
  };

  const toggleChunk = (chunkId: number) => {
    const newSelected = new Set(selectedChunks);
    if (newSelected.has(chunkId)) {
      newSelected.delete(chunkId);
    } else {
      newSelected.add(chunkId);
    }
    setSelectedChunks(newSelected);
  };

  const selectAll = () => {
    setSelectedChunks(new Set(filteredChunks.map(c => c.id)));
  };

  const clearSelection = () => {
    setSelectedChunks(new Set());
  };

  const toggleExpanded = (chunkId: number) => {
    const newExpanded = new Set(expandedChunks);
    if (newExpanded.has(chunkId)) {
      newExpanded.delete(chunkId);
    } else {
      newExpanded.add(chunkId);
    }
    setExpandedChunks(newExpanded);
  };

  const handleApply = () => {
    if (selectedChunks.size === 0) {
      toast.warning('No chunks selected', {
        description: 'Please select at least one chunk to use in your query'
      });
      return;
    }
    onApplySelection(Array.from(selectedChunks));
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-[95vw] sm:max-w-4xl max-h-[90vh] flex flex-col">
        <DialogHeader>
          <DialogTitle>View & Select Chunks</DialogTitle>
          <DialogDescription>
            {documentName} • {chunks.length} total chunks
          </DialogDescription>
        </DialogHeader>

        {/* Search and Actions Bar */}
        <div className="flex gap-2 items-center">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search chunks..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9"
            />
          </div>
          <Badge variant="secondary">
            {selectedChunks.size} selected
          </Badge>
          <Button
            variant="outline"
            size="sm"
            onClick={selectAll}
            disabled={loading}
          >
            Select All
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={clearSelection}
            disabled={loading || selectedChunks.size === 0}
          >
            Clear
          </Button>
        </div>

        {/* Chunks List */}
        <div className="flex-1 overflow-hidden">
          <ScrollArea className="h-full w-full">
            <div className="pr-4">
              {loading ? (
                <div className="flex items-center justify-center h-40">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary" />
                </div>
              ) : filteredChunks.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
              <Search className="h-12 w-12 mb-2" />
              <p>No chunks found</p>
              {searchQuery && (
                <p className="text-sm">Try a different search term</p>
              )}
            </div>
          ) : (
            <div className="space-y-3 pb-4">
              {filteredChunks.map((chunk) => (
                <Card
                  key={chunk.id}
                  className={`p-4 cursor-pointer transition-all hover:shadow-md ${
                    selectedChunks.has(chunk.id) ? 'ring-2 ring-primary bg-primary/5' : ''
                  }`}
                  onClick={() => toggleChunk(chunk.id)}
                >
                  <div className="flex items-start gap-3">
                    <Checkbox
                      checked={selectedChunks.has(chunk.id)}
                      onCheckedChange={() => toggleChunk(chunk.id)}
                      className="mt-1"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-2">
                        <Badge variant="outline" className="text-xs">
                          Chunk #{chunk.id + 1}
                        </Badge>
                        {chunk.metadata.title && (
                          <span className="text-xs text-muted-foreground truncate">
                            {chunk.metadata.title}
                          </span>
                        )}
                      </div>
                      <p className={`text-sm whitespace-pre-wrap break-words ${expandedChunks.has(chunk.id) ? '' : 'line-clamp-3'}`}>
                        {chunk.content}
                      </p>
                      {chunk.content.length > 200 && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleExpanded(chunk.id);
                          }}
                          className="mt-1 h-6 px-2 text-xs"
                        >
                          {expandedChunks.has(chunk.id) ? 'Show less' : 'Show more'}
                        </Button>
                      )}
                      {chunk.images && chunk.images.length > 0 && (
                        <div className="mt-3 space-y-2">
                          <div className="text-xs font-medium text-muted-foreground">Images:</div>
                          <div className="grid grid-cols-1 gap-2">
                            {chunk.images.map((img, imgIndex) => {
                              const backendUrl = (typeof window !== "undefined" && localStorage.getItem("backendUrl")) || "";
                              const imageSrc = backendUrl ? `${backendUrl.replace(/\/$/, "")}/image_bytes/${img.id}` : img.url;
                              return (
                                <div key={img.id} className="border rounded p-2 bg-muted/50">
                                  <img
                                    src={imageSrc}
                                    alt={`Image ${imgIndex + 1}`}
                                    className="max-w-full h-auto max-h-32 object-contain rounded"
                                  />
                                  <p className="text-xs mt-1 text-muted-foreground">{img.summary}</p>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                      {chunk.metadata.page && (
                        <div className="mt-2 text-xs text-muted-foreground">
                          Page {chunk.metadata.page}
                        </div>
                      )}
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}
            </div>
          </ScrollArea>
        </div>

        {/* Footer Actions */}
        <div className="flex justify-between items-center pt-4 border-t">
          <div className="text-sm text-muted-foreground">
            {selectedChunks.size > 0 && (
              <span>
                {selectedChunks.size} chunk{selectedChunks.size !== 1 ? 's' : ''} selected for query
              </span>
            )}
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={onClose}>
              <X className="h-4 w-4 mr-2" />
              Cancel
            </Button>
            <Button onClick={handleApply} disabled={selectedChunks.size === 0}>
              <Check className="h-4 w-4 mr-2" />
              Use Selected Chunks
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
