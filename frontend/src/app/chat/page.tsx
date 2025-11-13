import { ChatInterface } from '@/components/ChatInterface';

export default function ChatPage() {
  return (
    <div className="flex-1 w-full">
      <main className="flex-1 flex min-h-screen bg-background">
        <ChatInterface />
      </main>
    </div>
  );
}
