'use client';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';

interface FileUploadCardProps {
  files: File[];
  loading: boolean;
  onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onFileRemove: (index: number) => void;
  onSummarize: () => void;
}

export function FileUploadCard({
  files,
  loading,
  onFileChange,
  onFileRemove,
  onSummarize
}: FileUploadCardProps) {
  return (
    <Card className="p-6 space-y-4 backdrop-blur-sm dark:bg-background/95">
      <div className="flex items-center space-x-2">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={1.5}
          stroke="currentColor"
          className="w-5 h-5 text-primary"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m6.75 12l-3-3m0 0l-3 3m3-3v6m-1.5-15H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"
          />
        </svg>
        <h2 className="text-xl font-semibold">Upload Document</h2>
      </div>
      <div className="space-y-4">
        <Input 
          type="file" 
          accept=".pdf,.doc,.docx,.txt"
          onChange={onFileChange}
          className="border-dashed cursor-pointer hover:border-primary/50 transition-colors"
          multiple
        />
        {files.length > 0 && (
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">Selected files:</p>
            <div className="max-h-32 overflow-y-auto space-y-2">
              {files.map((file, index) => (
                <div key={index} className="flex items-center justify-between bg-muted/50 p-2 rounded-md">
                  <span className="text-sm truncate flex-1 pr-2">{file.name}</span>
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    onClick={() => onFileRemove(index)}
                    className="text-destructive hover:text-destructive hover:bg-destructive/10"
                  >
                    Remove
                  </Button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      <Button 
        onClick={onSummarize} 
        disabled={files.length === 0 || loading} 
        className="w-full"
        variant={files.length > 0 ? "default" : "secondary"}
      >
        {loading ? 'Summarizing...' : files.length > 0 ? `Summarize ${files.length} file${files.length > 1 ? 's' : ''}` : 'Select files'}
      </Button>
    </Card>
  );
}
