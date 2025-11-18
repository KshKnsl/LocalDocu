"use client";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { SignIn, SignUp } from "@clerk/nextjs";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Check, Rocket, Globe, SlidersHorizontal, Github } from "lucide-react";
import ThemeSwitcher from "@/components/ui/theme-switcher";
import { BackendConfigDialog } from "@/components/BackendConfig";

export default function Home() {
  const [mounted, setMounted] = useState(false);
  const router = useRouter();
  useEffect(() => { const t = setTimeout(() => setMounted(true), 10); return () => clearTimeout(t); }, []);
  useEffect(() => {
    router.push('/spaces');
  }, []);

  return (
    <main className="w-full flex-1 flex flex-col items-center justify-center bg-background relative">
      <div className="fixed top-6 right-6 z-50 flex items-center gap-2"><BackendConfigDialog /><ThemeSwitcher /></div>
      <section className={`mx-auto max-w-7xl px-4 py-12 md:py-20 transition-all duration-500 ease-out ${mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-3"}`}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
          <div className="flex flex-col items-start gap-6">
            <Badge className="w-fit inline-flex items-center gap-2" variant="secondary"><Check className="w-4 h-4" />100% Private · Fully Local · Zero Cloud</Badge>
            <h1 className="text-balance text-3xl font-semibold tracking-tight md:text-5xl leading-tight">
              <span className="block">Your Documents Stay Private.</span><span className="block">Runs Completely On Your Machine.</span>
            </h1>
            <p className="max-w-xl text-pretty text-base text-muted-foreground md:text-lg">A privacy-first document summarizer that runs entirely locally using Ollama. No cloud uploads, no external APIs, no data leaks. Everything processes on your own hardware with open-source AI models you control.</p>
            <div className="flex flex-col sm:flex-row gap-3 w-full">
              <a href="/spaces" className="w-full sm:w-auto"><Button size="lg" className="w-full sm:w-auto inline-flex items-center gap-2"><Rocket className="w-4 h-4" />Start Using App</Button></a>
              <a href="#signin" className="w-full sm:w-auto"><Button size="lg" variant="outline" className="w-full sm:w-auto">Sign In (Optional)</Button></a>
            </div>
          </div>
          <Card className="shadow-lg border-gray-100 dark:border-gray-800">
            <CardHeader><CardTitle className="text-base">Quick start</CardTitle><CardDescription>Get productive in a few clicks</CardDescription></CardHeader>
            <CardContent className="grid gap-4">
              <div className="flex items-start gap-3"><Check className="w-5 h-5 text-green-500 mt-1" /><div><p className="font-medium">100% Local Processing</p><p className="text-sm text-muted-foreground">All AI runs on your machine via Ollama. Zero cloud dependencies.</p></div></div>
              <div className="flex items-start gap-3"><Check className="w-5 h-5 text-green-500 mt-1" /><div><p className="font-medium">Scale to long documents</p><p className="text-sm text-muted-foreground">Chunking and map-reduce preserve document structure.</p></div></div>
              <div className="flex justify-end mt-2"><a href="#signin"><Button size="sm">Create account</Button></a></div>
            </CardContent>
          </Card>
        </div>
      </section>
      <section id="signin" className={`mx-auto max-w-7xl px-4 py-8 md:py-10 transition-all duration-300 ease-out ${mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-3"}`} style={{ transitionDelay: mounted ? "140ms" : undefined }}>
        <div className="mb-8 max-w-xl"><h2 className="text-2xl font-semibold tracking-tight md:text-3xl">Access your private workspace</h2><p className="text-muted-foreground">Sign in to start processing documents locally on your machine. All processing stays private.</p></div>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 items-start">
          <Card className="p-0 overflow-hidden">
            <div className="p-4">
              <div className="flex items-center justify-between gap-4"><div><h3 className="text-base font-semibold">Welcome</h3><p className="text-sm text-muted-foreground">Continue with your account</p></div></div>
              <div className="mt-4 space-y-3">
                <div className="flex items-start gap-3"><Rocket className="w-4 h-4 text-green-500 mt-1 flex-shrink-0" /><div className="text-sm"><p className="font-medium">Runs Locally</p><p className="text-xs text-muted-foreground">Powered by Ollama on your hardware</p></div></div>
                <div className="flex items-start gap-3"><Globe className="w-4 h-4 text-green-500 mt-1 flex-shrink-0" /><div className="text-sm"><p className="font-medium">Universal Access</p><p className="text-xs text-muted-foreground">Works with any document format</p></div></div>
                <div className="flex items-start gap-3"><SlidersHorizontal className="w-4 h-4 text-green-500 mt-1 flex-shrink-0" /><div className="text-sm"><p className="font-medium">Open Source Models</p><p className="text-xs text-muted-foreground">Use Mistral, Llama, or any Ollama model</p></div></div>
              </div>
              <div className="mt-6 flex flex-col sm:flex-row gap-3">
                <Dialog><DialogTrigger asChild><Button size="lg" className="flex-1 font-semibold px-4 py-3 bg-black text-white dark:bg-white dark:text-black">Sign In</Button></DialogTrigger>
                  <DialogContent className="sm:max-w-[435px] p-0 bg-white dark:bg-gray-900 border-gray-200 dark:border-gray-800 shadow-xl dark:shadow-gray-950/50">
                    <DialogHeader className="border-b border-gray-200 dark:border-gray-800"><DialogTitle className="px-4 pt-4 text-gray-900 dark:text-white">Welcome Back</DialogTitle></DialogHeader>
                    <div className="px-4 pb-4"><SignIn appearance={{ elements: { rootBox: "w-full", card: "w-full shadow-none p-0 bg-transparent", headerTitle: "text-gray-900 dark:text-white" } }} /></div>
                  </DialogContent>
                </Dialog>
                <Dialog><DialogTrigger asChild><Button size="lg" variant="outline" className="flex-1 px-4 py-3">Create Account</Button></DialogTrigger>
                  <DialogContent className="sm:max-w-[435px] p-0 bg-white dark:bg-gray-900 border-gray-200 dark:border-gray-800 shadow-xl dark:shadow-gray-950/50">
                    <DialogHeader className="border-b border-gray-200 dark:border-gray-800"><DialogTitle className="px-4 pt-4 text-gray-900 dark:text-white">Create Your Account</DialogTitle></DialogHeader>
                    <div className="px-4 pb-4"><SignUp appearance={{ elements: { rootBox: "w-full", card: "w-full shadow-none p-0 bg-transparent" } }} /></div>
                  </DialogContent>
                </Dialog>
              </div>
            </div>
          </Card>
          <Card className="p-4 mb-0"><CardHeader className="p-0 mb-0"><CardTitle className="text-base">Privacy-First Benefits</CardTitle><CardDescription>Why go local matters</CardDescription></CardHeader>
            <CardContent className="text-sm text-muted-foreground grid gap-3 mb-0">
              <div className="flex items-start gap-2"><Badge variant="outline" className="bg-green-50 dark:bg-green-950 border-green-500">Private</Badge><p className="m-0">Documents processed entirely on your hardware.</p></div>
              <div className="flex items-start gap-2"><Badge variant="outline" className="bg-green-50 dark:bg-green-950 border-green-500">Offline</Badge><p className="m-0">Works without internet. Perfect for sensitive or confidential documents.</p></div>
              <div className="flex items-start gap-2"><Badge variant="outline" className="bg-green-50 dark:bg-green-950 border-green-500">Open Source</Badge><p className="m-0">Uses Ollama with models like Mistral, Llama. Fully transparent AI.</p></div>
              <div className="flex items-start gap-2"><Badge variant="outline" className="bg-green-50 dark:bg-green-950 border-green-500">Free</Badge><p className="m-0">No API costs. No per-document fees. Unlimited processing on your machine.</p></div>
              <div className="flex items-start gap-2"><Badge variant="outline" className="bg-green-50 dark:bg-green-950 border-green-500">Control</Badge><p className="m-0">You own the models, data, and infrastructure. No vendor lock-in.</p></div>
              <div className="flex items-start gap-2"><Badge variant="outline" className="bg-green-50 dark:bg-green-950 border-green-500">Fast</Badge><p className="m-0">Local processing with GPU acceleration. No network latency.</p></div>
            </CardContent>
          </Card>
        </div>
      </section>
      <section id="capabilities" className={`mx-auto max-w-7xl px-4 py-8 md:py-10 transition-all duration-300 ease-out ${mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-3"}`} style={{ transitionDelay: mounted ? "80ms" : undefined }}>
        <div className="mb-6 max-w-xl"><h2 className="text-2xl font-semibold tracking-tight md:text-3xl">Privacy Meets Capability</h2><p className="text-muted-foreground">Why local processing solves real problems better than cloud solutions.</p></div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Accordion type="single" collapsible defaultValue="item-3" className="space-y-2">
            <AccordionItem value="item-1"><AccordionTrigger className="rounded-md border px-4 py-3 text-left">How does local processing work?</AccordionTrigger><AccordionContent className="text-sm text-muted-foreground p-4 border-x border-b rounded-b-md">We use Ollama to run open-source AI models (Mistral, Llama, Phi, etc.) directly on your machine. Documents are processed entirely locally with no cloud uploads. The AI runs on your CPU/GPU using optimized inference engines for fast results.</AccordionContent></AccordionItem>
            <AccordionItem value="item-2"><AccordionTrigger className="rounded-md border px-4 py-3 text-left">What reduces hallucinations?</AccordionTrigger><AccordionContent className="text-sm text-muted-foreground p-4 border-x border-b rounded-b-md">A multi-stage verification pass checks entities and claims against the source document, with optional knowledge graph integration to validate facts. We also use confidence scoring and uncertainty quantification to flag potentially unreliable content before final output.</AccordionContent></AccordionItem>
            <AccordionItem value="item-3"><AccordionTrigger className="rounded-md border px-4 py-3 text-left">How fast is the processing?</AccordionTrigger><AccordionContent className="text-sm text-muted-foreground p-4 border-x border-b rounded-b-md">Processing times vary by document length and complexity. Typical documents (10-50 pages) are summarized in 10-30 seconds. Very long documents (100+ pages) may take 2-5 minutes due to thorough analysis and verification steps.</AccordionContent></AccordionItem>
            <AccordionItem value="item-4"><AccordionTrigger className="rounded-md border px-4 py-3 text-left">What file formats are supported?</AccordionTrigger><AccordionContent className="text-sm text-muted-foreground p-4 border-x border-b rounded-b-md">We support PDF, Word (.docx), plain text, HTML, Markdown, and structured formats like JSON/XML. Image files are not summarized, but you can attach images in chat and ask questions about them.</AccordionContent></AccordionItem>
          </Accordion>
          <Accordion type="single" collapsible defaultValue="item-5" className="space-y-2">
            <AccordionItem value="item-5"><AccordionTrigger className="rounded-md border px-4 py-3 text-left">Do you support multimodal inputs?</AccordionTrigger><AccordionContent className="text-sm text-muted-foreground p-4 border-x border-b rounded-b-md">Partially. Document summarization is text-only. For images, you can attach them in chat and ask questions (Q&A) about the content. Audio/video transcripts can be summarized if provided as text.</AccordionContent></AccordionItem>
            <AccordionItem value="item-6"><AccordionTrigger className="rounded-md border px-4 py-3 text-left">How is quality evaluated?</AccordionTrigger><AccordionContent className="text-sm text-muted-foreground p-4 border-x border-b rounded-b-md">We apply multi-criteria evaluation beyond n-gram overlap, measuring semantics, faithfulness, coherence, and completeness. Hybrid Human-AI reviews are supported, with A/B testing capabilities for comparing different summarization approaches.</AccordionContent></AccordionItem>
            <AccordionItem value="item-7"><AccordionTrigger className="rounded-md border px-4 py-3 text-left">Is there an API available?</AccordionTrigger><AccordionContent className="text-sm text-muted-foreground p-4 border-x border-b rounded-b-md">Yes, we provide RESTful APIs with comprehensive documentation, SDKs for popular languages (Python, JavaScript, Java), webhook support for async processing, and batch processing capabilities for high-volume use cases.</AccordionContent></AccordionItem>
            <AccordionItem value="item-8"><AccordionTrigger className="rounded-md border px-4 py-3 text-left">What about data privacy and security?</AccordionTrigger><AccordionContent className="text-sm text-muted-foreground p-4 border-x border-b rounded-b-md">All documents are encrypted in transit and at rest using AES-256. We offer on-premise deployment options, GDPR compliance, and data retention policies. Documents are never stored longer than necessary for processing.</AccordionContent></AccordionItem>
          </Accordion>
        </div>
      </section>
      <section className={`mx-auto max-w-7xl px-4 py-8 border-t border-border/50 transition-all duration-300 ease-out ${mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-3"}`} style={{ transitionDelay: mounted ? "100ms" : undefined }}>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 text-center sm:text-left">
          <div className="flex items-center gap-2">
            <Github className="w-5 h-5" />
            <span className="text-sm font-medium">Open Source</span>
          </div>
          <p className="text-sm text-muted-foreground">
            This project is open source and available on GitHub. 
            <a 
              href="https://github.com/KshKnsl/minorproject" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-primary hover:underline ml-1"
            >
              View on GitHub →
            </a>
          </p>
        </div>
      </section>
    </main>
  );
}