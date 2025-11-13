"use client";

import React from "react";
import { MessageSquare } from "lucide-react";

export function ChatEmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-[calc(100vh-10rem)] p-4">
      <div className="max-w-2xl w-full space-y-6">
        <div className="text-center space-y-2">
          <MessageSquare className="h-12 w-12 text-primary mx-auto mb-2" />
          <h2 className="text-2xl font-semibold">Document Summarizer</h2>
          <p className="text-sm text-muted-foreground">
            Upload documents and get intelligent summaries, analysis, and answers to your questions. 
            Perfect for research assistance and beyond.
          </p>
        </div>
      </div>
    </div>
  );
}
