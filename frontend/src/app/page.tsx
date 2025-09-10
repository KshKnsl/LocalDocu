'use client';
import { useState } from 'react';
import { useUser } from '@clerk/nextjs';
import { SignInCard } from '@/components/SignInCard';
import { FileUploadCard } from '@/components/FileUploadCard';
import { ChatCard } from '@/components/ChatCard';
import { SummaryCard } from '@/components/SummaryCard';

export default function Home() {
  const { isSignedIn } = useUser();
  const [files, setFiles] = useState<File[]>([]);
  const [chat, setChat] = useState('');
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files);
      setFiles(prevFiles => [...prevFiles, ...newFiles]);
    }
  };

  const handleSummarize = async () => {
    if (files.length === 0) return;

    setLoading(true);
    setSummary('');
    
    try {
      const formData = new FormData();
      files.forEach((file, index) => {
        formData.append(`file${index}`, file);
      });

      const response = await fetch('/api/summarize', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Error summarizing document');
      }

      setSummary(data.summary);
    } catch (error) {
      console.error('Error:', error);
      setSummary('Error processing your document. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleChat = async () => {
    if (!chat.trim()) return;

    setLoading(true);
    const userMessage = chat;
    setChat('');
    
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userMessage }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Error processing chat message');
      }

      setSummary(data.response);
    } catch (error) {
      console.error('Error:', error);
      setSummary('Error processing your message. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  if (!isSignedIn) {
    return <SignInCard />;
  }

  return (
    <div className="container mx-auto max-w-4xl py-10 space-y-8">
      <div className="grid gap-8 md:grid-cols-2">
        <FileUploadCard
          files={files}
          loading={loading}
          onFileChange={handleFile}
          onFileRemove={(index) => setFiles(prev => prev.filter((_, i) => i !== index))}
          onSummarize={handleSummarize}
        />
        <ChatCard
          chat={chat}
          loading={loading}
          onChatChange={(e) => setChat(e.target.value)}
          onSendMessage={handleChat}
        />
      </div>
      <SummaryCard summary={summary} />
    </div>
  );
}
