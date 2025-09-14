'use client';
import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function SsoCallback() {
  const router = useRouter();
  useEffect(() => {
    // Clerk will handle the callback, you can redirect or show a loading state
    router.replace('/');
  }, [router]);
  return <div>Signing you in...</div>;
}
