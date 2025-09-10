'use client';
import { SignIn } from '@clerk/nextjs';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

export function SignInCard() {
  return (
    <div className="flex min-h-[80vh] flex-col items-center justify-center px-4">
      <Card className="w-full max-w-md p-8 text-center backdrop-blur-sm dark:bg-background/95">
        <h2 className="text-3xl font-bold tracking-tight mb-2">Welcome to AI Summarizer</h2>
        <p className="text-muted-foreground mb-8">Upload documents and chat with our AI to get instant summaries.</p>
        <Dialog>
          <DialogTrigger asChild>
            <Button size="lg" className="font-semibold">
              Get Started
            </Button>
          </DialogTrigger>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Sign In</DialogTitle>
            </DialogHeader>
            <div className="py-4">
              <SignIn />
            </div>
          </DialogContent>
        </Dialog>
      </Card>
    </div>
  );
}
