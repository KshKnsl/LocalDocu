"use client";

import { Progress } from "@/components/ui/progress";
import { FileText, Loader2, CheckCircle2, AlertCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface ProcessingFile {
  name: string;
  status: 'uploading' | 'processing' | 'done' | 'failed';
  progress?: number;
  chunks?: number;
  currentChunk?: number;
}

interface ProcessingBannerProps {
  files: ProcessingFile[];
  onDismiss?: () => void;
}

export function ProcessingBanner({ files, onDismiss }: ProcessingBannerProps) {
  if (files.length === 0) return null;

  const activeFiles = files.filter(f => f.status !== 'done' && f.status !== 'failed');
  const completedFiles = files.filter(f => f.status === 'done');
  const failedFiles = files.filter(f => f.status === 'failed');
  
  const totalProgress = files.length > 0 
    ? files.reduce((sum, f) => sum + (f.progress || 0), 0) / files.length 
    : 0;

  const allComplete = activeFiles.length === 0;

  return (
    <div className="space-y-4">
        {!allComplete && (
          <div className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Overall Progress</span>
              <span className="font-medium">{Math.round(totalProgress)}%</span>
            </div>
            <Progress value={totalProgress} className="h-2" />
          </div>
        )}

        <div className="flex gap-2 flex-wrap">
          {activeFiles.length > 0 && (
            <Badge variant="outline" className="bg-blue-50 dark:bg-blue-950 border-blue-500 text-blue-700 dark:text-blue-300">
              <Loader2 className="w-3 h-3 mr-1 animate-spin" />
              {activeFiles.length} Processing
            </Badge>
          )}
          {completedFiles.length > 0 && (
            <Badge variant="outline" className="bg-green-50 dark:bg-green-950 border-green-500 text-green-700 dark:text-green-300">
              <CheckCircle2 className="w-3 h-3 mr-1" />
              {completedFiles.length} Complete
            </Badge>
          )}
          {failedFiles.length > 0 && (
            <Badge variant="outline" className="bg-red-50 dark:bg-red-950 border-red-500 text-red-700 dark:text-red-300">
              <AlertCircle className="w-3 h-3 mr-1" />
              {failedFiles.length} Failed
            </Badge>
          )}
        </div>

        <div className="space-y-2 max-h-[300px] overflow-y-auto">
          {files.map((file, idx) => (
            <div 
              key={idx}
              className="flex items-start gap-3 p-2 rounded-md bg-muted/50 hover:bg-muted/70 transition-colors"
            >
              <FileText className={`w-4 h-4 mt-0.5 flex-shrink-0 ${
                file.status === 'done' ? 'text-green-500' :
                file.status === 'failed' ? 'text-red-500' :
                'text-blue-500'
              }`} />
              
              <div className="flex-1 min-w-0 space-y-1">
                <div className="flex items-center justify-between gap-2">
                  <p className="text-sm font-medium truncate">{file.name}</p>
                  {file.status === 'done' && (
                    <CheckCircle2 className="w-4 h-4 text-green-500 flex-shrink-0" />
                  )}
                  {file.status === 'failed' && (
                    <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
                  )}
                </div>
                
                {file.status === 'uploading' && (
                  <p className="text-xs text-muted-foreground">Uploading...</p>
                )}
                
                {file.status === 'processing' && (
                  <>
                    {file.chunks && (
                      <p className="text-xs text-muted-foreground">
                        Processing chunk {file.currentChunk || 1} of {file.chunks}
                      </p>
                    )}
                    {!file.chunks && (
                      <p className="text-xs text-muted-foreground">
                        Analyzing document structure...
                      </p>
                    )}
                    {file.progress !== undefined && (
                      <Progress value={file.progress} className="h-1" />
                    )}
                  </>
                )}
                
                {file.status === 'done' && (
                  <p className="text-xs text-green-600 dark:text-green-400">
                    Ready for RAG queries
                  </p>
                )}
                
                {file.status === 'failed' && (
                  <p className="text-xs text-red-600 dark:text-red-400">
                    Processing failed
                  </p>
                )}
              </div>
            </div>
          ))}
        </div>
    </div>
  );
}
