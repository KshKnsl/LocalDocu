import { useUser } from '@clerk/nextjs';
import { SignInCard } from '@/components/SignInCard';
import { ChatInterface } from '@/components/ChatInterface';

export default function Home() {
  const { isSignedIn } = useUser();

  if (!isSignedIn) {
    return <SignInCard />;
  }

  return (
    <div className="flex-1 w-full">
      <main className="flex-1 flex min-h-screen bg-background">
        <ChatInterface />
      </main>
    </div>
  );
}