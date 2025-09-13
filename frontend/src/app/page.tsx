'use client';
import { SignInCard } from '@/components/SignInCard';
import { ChatInterface } from '@/components/ChatInterface';
import { useUser } from '@clerk/nextjs';

function HomeContent() {
  const { isSignedIn } = useUser();
  if (!isSignedIn) return <SignInCard />;
  return (
    <div className="flex-1 w-full">
      <main className="flex-1 flex min-h-screen bg-background">
        <ChatInterface />
      </main>
    </div>
  );
}

export default function Home() {
  return <HomeContent />;
}