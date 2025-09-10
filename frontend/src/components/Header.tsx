'use client';

import {
  SignIn,
  SignUp,
  UserButton,
  useUser
} from '@clerk/nextjs';
import Link from 'next/link';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "./ui/button";
import Image from 'next/image';
import ThemeSwitcher from './Theme-switcher';

export default function Header() {
  const { isSignedIn, user } = useUser();

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur-sm">
      <div className="container flex h-14 items-center">
        <Link href="/" className="flex items-center space-x-3">
          <div className="relative h-8 w-8">
            <Image
              src="/logo.png"
              alt="AI Summarizer Logo"
              width={32}
              height={32}
              className="object-contain"
              priority
            />
          </div>
          <span className="font-bold">AI Summarizer</span>
        </Link>
        <div className="flex flex-1 items-center justify-end space-x-4">
          <nav className="flex items-center space-x-2">
             <ThemeSwitcher />
            {!isSignedIn ? (
              <>
                <Dialog>
                  <DialogTrigger asChild>
                    <Button>Sign In</Button>
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
                <Dialog>
                  <DialogTrigger asChild>
                    <Button>Sign Up</Button>
                  </DialogTrigger>
                  <DialogContent className="sm:max-w-[425px]">
                    <DialogHeader>
                      <DialogTitle>Sign Up</DialogTitle>
                    </DialogHeader>
                    <div className="py-4">
                      <SignUp />
                    </div>
                  </DialogContent>
                </Dialog>
              </>
            ) : (
              <>
                <Button>
                  <Link href="/profile">
                    {user?.firstName ? `Hi, ${user.firstName}` : 'Profile'}
                  </Link>
                </Button>
                <UserButton afterSignOutUrl="/" />
              </>
            )}
          </nav>
        </div>
      </div>
    </header>
  );
}
