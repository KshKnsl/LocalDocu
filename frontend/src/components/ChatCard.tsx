'use client';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card } from '@/components/ui/card';

interface ChatCardProps {
  chat: string;
  loading: boolean;
  onChatChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  onSendMessage: () => void;
}

export function ChatCard({
  chat,
  loading,
  onChatChange,
  onSendMessage
}: ChatCardProps) {
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
            d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z"
          />
        </svg>
        <h2 className="text-xl font-semibold">Chat with AI</h2>
      </div>
      <Textarea
        value={chat}
        onChange={onChatChange}
        placeholder="Ask about your document..."
        rows={3}
        className="resize-none"
      />
      <Button 
        onClick={onSendMessage} 
        disabled={!chat || loading} 
        className="w-full"
        variant={chat ? "default" : "secondary"}
      >
        {loading ? 'Sending...' : 'Send Message'}
      </Button>
    </Card>
  );
}
