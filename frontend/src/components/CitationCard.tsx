"use client";

import React, { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { FileText, TrendingUp } from "lucide-react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import type { Citation } from "@/lib/api";

interface CitationCardProps {
  citation: Citation;
}

export function CitationCard({ citation }: CitationCardProps) {
  const [showFull, setShowFull] = useState(false);

  const scorePercent = citation.score !== undefined 
    ? `${(citation.score * 100).toFixed(0)}%` 
    : null;

  return (
    <>
      <Badge
        variant="outline"
        className="cursor-pointer hover:bg-accent transition-colors flex items-center gap-1.5 px-2 py-1"
        onClick={() => setShowFull(true)}
      >
        <FileText className="h-3 w-3" />
        <span className="text-xs">
          Page {citation.page}
        </span>
        {scorePercent && (
          <>
            <span className="text-muted-foreground">•</span>
            <span className="text-xs font-medium text-primary flex items-center gap-0.5">
              <TrendingUp className="h-3 w-3" />
              {scorePercent}
            </span>
          </>
        )}
      </Badge>

      <Dialog open={showFull} onOpenChange={setShowFull}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>Citation Details</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-xs text-muted-foreground mb-1">Page</div>
                <div className="text-sm font-semibold">{citation.page}</div>
              </div>
              {citation.score !== undefined && (
                <div>
                  <div className="text-xs text-muted-foreground mb-1">Relevance Score</div>
                  <div className="text-sm font-semibold flex items-center gap-1.5">
                    <TrendingUp className="h-4 w-4 text-primary" />
                    <span className="text-primary">{scorePercent}</span>
                    <span className="text-xs text-muted-foreground font-normal">
                      (Rank #{citation.rank})
                    </span>
                  </div>
                </div>
              )}
            </div>
            <div>
              <div className="text-xs text-muted-foreground mb-1">Document ID</div>
              <div className="text-sm font-mono bg-muted px-2 py-1 rounded">{citation.documentId}</div>
            </div>
            {citation.source && citation.source !== citation.documentId && (
              <div>
                <div className="text-xs text-muted-foreground mb-1">Source</div>
                <div className="text-sm bg-muted px-2 py-1 rounded">{citation.source}</div>
              </div>
            )}
            <div>
              <div className="text-xs text-muted-foreground mb-1">Source Text</div>
              <div className="max-h-[50vh] overflow-y-auto bg-muted p-3 rounded text-sm whitespace-pre-wrap">
                {citation.fullText}
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
