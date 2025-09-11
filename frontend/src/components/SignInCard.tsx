"use client";
import { SignIn } from "@clerk/nextjs";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

export function SignInCard() {
  return (
    <div className="w-full min-h-[calc(100vh-4rem)] flex flex-col items-center justify-center relative bg-white dark:bg-gray-950 transition-colors duration-300">
      {/* Background decoration */}
      <div className="absolute inset-0 -z-10 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-50/80 via-white to-blue-50/80 dark:from-gray-900 dark:via-gray-900/50 dark:to-gray-900 transition-colors duration-300" />
        <div className="absolute inset-0 bg-grid-black/[0.02] dark:bg-grid-white/[0.02] transition-opacity duration-300" />
        <div className="absolute inset-y-0 right-0 w-1/2 bg-gradient-to-l from-blue-100/20 to-transparent dark:from-blue-950/20 blur-3xl transform translate-x-1/3 transition-colors duration-300" />
        <div className="absolute inset-y-0 left-0 w-1/2 bg-gradient-to-r from-blue-100/20 to-transparent dark:from-blue-950/20 blur-3xl transform -translate-x-1/3 transition-colors duration-300" />
      </div>

      {/* Main content */}
      <div className="w-full max-w-7xl px-4 py-12 sm:px-6 lg:py-16 relative z-10">
        <div className="space-y-12">
          {/* Hero section */}
          <div className="text-center space-y-6">
            <h1 className="text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-blue-400 dark:to-blue-200 transition-colors duration-300">
                AI Summarizer
              </span>
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto transition-colors duration-300">
              Transform your documents into actionable insights with AI-powered
              summaries
            </p>
          </div>

          {/* Features */}
          <div className="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-3 text-center">
            {[
              {
                title: "Quick Summaries",
                description:
                  "Get instant summaries of your documents, powered by advanced AI",
                icon: "📄",
              },
              {
                title: "Interactive Chat",
                description:
                  "Ask questions and get detailed insights about your documents",
                icon: "💬",
              },
              {
                title: "Multiple Formats",
                description: "Support for PDF, Word, and text documents",
                icon: "📁",
              },
            ].map((feature) => (
              <Card
                key={feature.title}
                className="group p-6 backdrop-blur-sm bg-white/50 dark:bg-gray-800/50 border border-gray-200 dark:border-gray-800 hover:border-blue-200 dark:hover:border-blue-800 shadow-sm hover:shadow-md dark:shadow-gray-900/30 transition-all duration-300"
              >
                <div className="text-4xl mb-4 transform group-hover:scale-110 transition-transform duration-300">
                  {feature.icon}
                </div>
                <h3 className="text-lg font-semibold mb-2 text-gray-900 dark:text-white transition-colors duration-300">
                  {feature.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-300 transition-colors duration-300">
                  {feature.description}
                </p>
              </Card>
            ))}
          </div>

          {/* CTA */}
          <div className="text-center">
            <Dialog>
              <DialogTrigger asChild>
                <Button
                  size="lg"
                  className="font-semibold px-8 py-6 text-lg bg-gradient-to-r from-blue-500 to-blue-600 dark:from-blue-600 dark:to-blue-500 hover:from-blue-600 hover:to-blue-700 dark:hover:from-blue-500 dark:hover:to-blue-400 shadow-lg hover:shadow-xl dark:shadow-blue-950/20 transition-all duration-300 transform hover:scale-[1.02] active:scale-[0.98]"
                >
                  Get Started Now
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-[435px] p-0 bg-white dark:bg-gray-900 border-gray-200 dark:border-gray-800 shadow-xl dark:shadow-gray-950/50">
                <DialogHeader className="border-b border-gray-200 dark:border-gray-800">
                  <DialogTitle className="px-4 pt-4 text-gray-900 dark:text-white">
                    Welcome to AI Summarizer
                  </DialogTitle>
                </DialogHeader>
                <div className="px-4 pb-4">
                  <SignIn
                    appearance={{
                      elements: {
                        rootBox: "w-full",
                        card: "w-full shadow-none p-0 bg-transparent",
                        headerTitle: "text-gray-900 dark:text-white",
                        headerSubtitle: "text-gray-500 dark:text-gray-400",
                        formButtonPrimary:
                          "bg-blue-500 dark:bg-blue-600 hover:bg-blue-600 dark:hover:bg-blue-500 text-white",
                        formFieldInput:
                          "bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700",
                        formFieldLabel: "text-gray-700 dark:text-gray-300",
                        footer: "text-gray-600 dark:text-gray-400",
                        footerActionLink:
                          "text-blue-500 dark:text-blue-400 hover:text-blue-600 dark:hover:text-blue-300",
                      },
                    }}
                  />
                </div>
              </DialogContent>
            </Dialog>
          </div>
        </div>
      </div>
    </div>
  );
}
