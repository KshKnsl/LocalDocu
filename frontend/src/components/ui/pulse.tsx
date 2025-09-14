import React from "react";

export function PulseLoader({ className = "" }: { className?: string }) {
  return (
    <span
      className={`inline-block w-3 h-3 rounded-full bg-primary animate-pulse ${className}`}
      aria-label="Loading"
    />
  );
}
