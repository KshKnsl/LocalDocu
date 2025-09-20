"use client"

import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { SignIn, SignUp } from "@clerk/nextjs"
import { useEffect, useState } from "react"
import { Check, Rocket, AlertTriangle, Layers, Zap, Globe, BarChart2, FileText, ToolCase, SlidersHorizontal } from "lucide-react"
import ThemeSwitcher from "./Theme-switcher"

export default function HomePage() {
  const [mounted, setMounted] = useState(false)
  useEffect(() => {
    const t = setTimeout(() => setMounted(true), 10)
    return () => clearTimeout(t)
  }, [])
  return (
    <main className="w-full flex-1 flex flex-col items-center justify-center bg-background relative">
      {/* Theme Switcher - Floating */}
      <div className="fixed top-6 right-6 z-50">
        <ThemeSwitcher />
      </div>
      
      <section
        className={`mx-auto max-w-7xl px-4 py-12 md:py-20 transition-all duration-500 ease-out ${
          mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-3"
        }`}
      >
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
          {/* Left: Hero copy */}
          <div className="flex flex-col items-start gap-6">
            <Badge className="w-fit inline-flex items-center gap-2" variant="secondary">
              <Check className="w-4 h-4" />
              Built to solve real summarisation problems
            </Badge>
            <h1 className="text-balance text-3xl font-semibold tracking-tight md:text-5xl leading-tight">
              <span className="block">Tackle token limits, “lost-in-the-middle,”</span>
              <span className="block">and hallucinations head-on</span>
            </h1>
            <p className="max-w-xl text-pretty text-base text-muted-foreground md:text-lg">
              A research-grade document summariser engineered for long documents and high accuracy—using hierarchical
              pipelines, factual verification, and multi-criteria evaluation. Perfect for research assistance and beyond.
            </p>

            <div className="flex flex-col sm:flex-row gap-3 w-full">
              <a href="#signin" className="w-full sm:w-auto">
                <Button size="lg" className="w-full sm:w-auto inline-flex items-center gap-2">
                  <Rocket className="w-4 h-4" />
                  Sign in to start
                </Button>
              </a>

              <a href="#problems" className="w-full sm:w-auto">
                <Button size="lg" variant="outline" className="w-full sm:w-auto">
                  See the problems we solve
                </Button>
              </a>
            </div>
          </div>

          {/* Right: CTA Card */}
          <Card className="shadow-lg border-gray-100 dark:border-gray-800">
            <CardHeader>
              <CardTitle className="text-base">Quick start</CardTitle>
              <CardDescription>Get productive in a few clicks</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <div className="flex items-start gap-3">
                <Check className="w-5 h-5 text-blue-500 mt-1" />
                <div>
                  <p className="font-medium">Accuracy-first pipelines</p>
                  <p className="text-sm text-muted-foreground">Fact-checking and hierarchical synthesis built in.</p>
                </div>
              </div>

              <div className="flex items-start gap-3">
                <Check className="w-5 h-5 text-blue-500 mt-1" />
                <div>
                  <p className="font-medium">Scale to long documents</p>
                  <p className="text-sm text-muted-foreground">Chunking and map-reduce preserve document structure.</p>
                </div>
              </div>

              <div className="flex justify-end mt-2">
                <a href="#signin">
                  <Button size="sm">Create account</Button>
                </a>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Sign In Section */}
      <section
        id="signin"
        className={`mx-auto max-w-7xl px-4 py-8 md:py-10 transition-all duration-300 ease-out ${
          mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-3"
        }`}
        style={{ transitionDelay: mounted ? ("140ms" as string) : undefined }}
      >
          <div className="mb-8 max-w-xl">
            <h2 className="text-2xl font-semibold tracking-tight md:text-3xl">Access your workspace</h2>
            <p className="text-muted-foreground">
              Sign in or create an account to generate accurate, verifiable summaries.
            </p>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 items-start">
          <Card className="p-0 overflow-hidden">
            <div className="p-4">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <h3 className="text-base font-semibold">Welcome</h3>
                  <p className="text-sm text-muted-foreground">Continue with your account</p>
                </div>
              </div>

              <div className="mt-4 space-y-3">
                <div className="flex items-start gap-3">
                  <Rocket className="w-4 h-4 text-blue-500 mt-1 flex-shrink-0" />
                  <div className="text-sm">
                    <p className="font-medium">Quick Setup</p>
                    <p className="text-xs text-muted-foreground">Get started in under 30 seconds</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <Globe className="w-4 h-4 text-purple-500 mt-1 flex-shrink-0" />
                  <div className="text-sm">
                    <p className="font-medium">Universal Access</p>
                    <p className="text-xs text-muted-foreground">Works with any document format</p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <SlidersHorizontal className="w-4 h-4 text-orange-500 mt-1 flex-shrink-0" />
                  <div className="text-sm">
                    <p className="font-medium">Custom Settings</p>
                    <p className="text-xs text-muted-foreground">Tailor output to your needs</p>
                  </div>
                </div>
              </div>

              <div className="mt-6 flex flex-col sm:flex-row gap-3">
                <Dialog>
                  <DialogTrigger asChild>
                    <Button size="lg" className="flex-1 font-semibold px-4 py-3 bg-black text-white dark:bg-white dark:text-black">
                      Sign In
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="sm:max-w-[435px] p-0 bg-white dark:bg-gray-900 border-gray-200 dark:border-gray-800 shadow-xl dark:shadow-gray-950/50">
                    <DialogHeader className="border-b border-gray-200 dark:border-gray-800">
                      <DialogTitle className="px-4 pt-4 text-gray-900 dark:text-white">Welcome Back</DialogTitle>
                    </DialogHeader>
                    <div className="px-4 pb-4">
                      <SignIn
                        appearance={{
                          elements: {
                            rootBox: "w-full",
                            card: "w-full shadow-none p-0 bg-transparent",
                            headerTitle: "text-gray-900 dark:text-white",
                          },
                        }}
                      />
                    </div>
                  </DialogContent>
                </Dialog>

                <Dialog>
                  <DialogTrigger asChild>
                    <Button size="lg" variant="outline" className="flex-1 px-4 py-3">
                      Create Account
                    </Button>
                  </DialogTrigger>
                  <DialogContent className="sm:max-w-[435px] p-0 bg-white dark:bg-gray-900 border-gray-200 dark:border-gray-800 shadow-xl dark:shadow-gray-950/50">
                    <DialogHeader className="border-b border-gray-200 dark:border-gray-800">
                      <DialogTitle className="px-4 pt-4 text-gray-900 dark:text-white">Create Your Account</DialogTitle>
                    </DialogHeader>
                    <div className="px-4 pb-4">
                      <SignUp
                        appearance={{
                          elements: {
                            rootBox: "w-full",
                            card: "w-full shadow-none p-0 bg-transparent",
                          },
                        }}
                      />
                    </div>
                  </DialogContent>
                </Dialog>
              </div>

            </div>
          </Card>

          <Card className="p-4 mb-0">
            <CardHeader className="p-0 mb-0">
              <CardTitle className="text-base">Why create an account?</CardTitle>
              <CardDescription>Problem-led benefits</CardDescription>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground grid gap-3 mb-0">
              <div className="flex items-start gap-2">
                <Badge variant="outline">Accuracy</Badge>
                <p className="m-0">Factual verification reduces hallucinations and preserves trust.</p>
              </div>
              <div className="flex items-start gap-2">
                <Badge variant="outline">Scale</Badge>
                <p className="m-0">Hierarchical pipelines handle 100k+ tokens without losing structure.</p>
              </div>
              <div className="flex items-start gap-2">
                <Badge variant="outline">Evaluation</Badge>
                <p className="m-0">Multi-criteria scoring compares methods for better quality.</p>
              </div>
              <div className="flex items-start gap-2">
                <Badge variant="outline">Speed</Badge>
                <p className="m-0">Optimized processing pipelines deliver results in seconds, not minutes.</p>
              </div>
              <div className="flex items-start gap-2">
                <Badge variant="outline">Security</Badge>
                <p className="m-0">Enterprise-grade encryption keeps your documents safe and private.</p>
              </div>
              <div className="flex items-start gap-2">
                <Badge variant="outline">Integration</Badge>
                <p className="m-0">API access and export formats for seamless workflow integration.</p>
              </div>
            </CardContent>
          </Card>
          </div>
        
      </section>

      {/* Merged: Challenges & Capabilities (compact) */}
      <section
        id="capabilities"
        className={`mx-auto max-w-7xl px-4 py-8 md:py-10 transition-all duration-300 ease-out ${
          mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-3"
        }`}
        style={{ transitionDelay: mounted ? ("80ms" as string) : undefined }}
      >
          <div className="mb-6 max-w-xl">
            <h2 className="text-2xl font-semibold tracking-tight md:text-3xl">Challenges & Capabilities</h2>
            <p className="text-muted-foreground">Compact view of common problems and our capability map.</p>
          </div>

          <div className="space-y-8">
            {/* Challenges Section */}
            <div>
              <h3 className="text-lg font-semibold mb-4 text-rose-600">Key Challenges</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {[
                  { title: "Token Limits", body: "Truncation across long documents; lost cross-section references.", icon: AlertTriangle },
                  { title: "Lost in the Middle", body: "Salient mid-document content gets ignored as prompts grow.", icon: Layers },
                  { title: "Hallucinations", body: "Ungrounded claims without verification.", icon: AlertTriangle },
                  { title: "Context Loss", body: "Important relationships between sections get disconnected.", icon: Globe },
                  { title: "Inconsistent Quality", body: "Variable output quality without proper evaluation metrics.", icon: BarChart2 },
                  { title: "Processing Speed", body: "Slow traditional methods can't handle real-time demands.", icon: SlidersHorizontal },
                ].map((c) => (
                  <div key={c.title} className="flex items-start gap-3 rounded-md border p-4">
                    <div className="flex-shrink-0 mt-1">
                      <c.icon className="w-5 h-5 text-rose-500" />
                    </div>
                    <div>
                      <div className="text-sm font-semibold">{c.title}</div>
                      <div className="text-xs text-muted-foreground">{c.body}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Capabilities Section */}
            <div>
              <h3 className="text-lg font-semibold mb-4 text-blue-600">Our Solutions</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                {[
                  { title: "Hierarchical Processing", body: "Map-reduce pipelines with recursive synthesis for long documents", icon: Layers },
                  { title: "Factual Verification", body: "Source-grounded checks with knowledge graph validation", icon: Zap },
                  { title: "Multi-Criteria Evaluation", body: "Semantics, faithfulness, coherence, and completeness scoring", icon: BarChart2 },
                  { title: "Multimodal Support", body: "Text, images, audio, and video integration for comprehensive analysis", icon: FileText },
                  { title: "Real-time Processing", body: "Optimized algorithms deliver results in seconds with parallel processing", icon: SlidersHorizontal },
                  { title: "Custom Models", body: "Fine-tuned models for domain-specific summarization tasks", icon: ToolCase },
                  { title: "API Integration", body: "RESTful APIs with webhooks for seamless workflow automation", icon: Globe },
                  { title: "Export Formats", body: "Multiple output formats: PDF, Word, HTML, JSON, and structured data", icon: FileText },
                ].map((cap) => (
                  <div key={cap.title} className="flex items-start gap-3 rounded-md border p-4">
                    <div className="flex-shrink-0 mt-1">
                      <cap.icon className="w-5 h-5 text-blue-500" />
                    </div>
                    <div>
                      <div className="text-sm font-semibold">{cap.title}</div>
                      <div className="text-xs text-muted-foreground">{cap.body}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        
      </section>

      {/* FAQ */}
      <section
        id="faq"
        className={`mx-auto max-w-7xl px-4 py-8 md:py-12 transition-all duration-500 ease-out ${
          mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-3"
        }`}
        style={{ transitionDelay: mounted ? ("320ms" as string) : undefined }}
      >
          <div className="mb-8 max-w-xl">
            <h2 className="text-2xl font-semibold tracking-tight md:text-3xl">Frequently Asked Questions</h2>
            <p className="text-muted-foreground">Answers to common questions about accuracy, speed, and scale.</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Accordion type="single" collapsible defaultValue="item-3" className="space-y-2">
              <AccordionItem value="item-1">
                <AccordionTrigger className="rounded-md border px-4 py-3 text-left">How do you handle extremely long documents?</AccordionTrigger>
                <AccordionContent className="text-sm text-muted-foreground p-4 border-x border-b rounded-b-md">
                  We use hierarchical chunking and Map-Reduce pipelines to preserve cross-section context and synthesise
                  coherent summaries well beyond typical token limits. Our system can process documents up to 1M+ tokens
                  while maintaining narrative flow and key relationships between sections.
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="item-2">
                <AccordionTrigger className="rounded-md border px-4 py-3 text-left">What reduces hallucinations?</AccordionTrigger>
                <AccordionContent className="text-sm text-muted-foreground p-4 border-x border-b rounded-b-md">
                  A multi-stage verification pass checks entities and claims against the source document, with optional knowledge graph
                  integration to validate facts. We also use confidence scoring and uncertainty quantification to flag potentially
                  unreliable content before final output.
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="item-3">
                <AccordionTrigger className="rounded-md border px-4 py-3 text-left">How fast is the processing?</AccordionTrigger>
                <AccordionContent className="text-sm text-muted-foreground p-4 border-x border-b rounded-b-md">
                  Processing times vary by document length and complexity. Typical documents (10-50 pages) are summarized in 10-30 seconds.
                  Very long documents (100+ pages) may take 2-5 minutes due to thorough analysis and verification steps.
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="item-4">
                <AccordionTrigger className="rounded-md border px-4 py-3 text-left">What file formats are supported?</AccordionTrigger>
                <AccordionContent className="text-sm text-muted-foreground p-4 border-x border-b rounded-b-md">
                  We support PDF, Word (.docx), plain text, HTML, Markdown, and structured formats like JSON/XML. 
                  Images and tables within documents are analyzed and incorporated into summaries where relevant.
                </AccordionContent>
              </AccordionItem>
            </Accordion>
            <Accordion type="single" collapsible defaultValue="item-5" className="space-y-2">
              <AccordionItem value="item-5">
                <AccordionTrigger className="rounded-md border px-4 py-3 text-left">Do you support multimodal inputs?</AccordionTrigger>
                <AccordionContent className="text-sm text-muted-foreground p-4 border-x border-b rounded-b-md">
                  Yes—text, images, audio transcripts, and video can be integrated so figures, diagrams, and multimedia content
                  contribute to comprehensive summaries. Charts and graphs are analyzed for key insights and data trends.
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="item-6">
                <AccordionTrigger className="rounded-md border px-4 py-3 text-left">How is quality evaluated?</AccordionTrigger>
                <AccordionContent className="text-sm text-muted-foreground p-4 border-x border-b rounded-b-md">
                  We apply multi-criteria evaluation beyond n-gram overlap, measuring semantics, faithfulness, coherence,
                  and completeness. Hybrid Human-AI reviews are supported, with A/B testing capabilities for comparing
                  different summarization approaches.
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="item-7">
                <AccordionTrigger className="rounded-md border px-4 py-3 text-left">Is there an API available?</AccordionTrigger>
                <AccordionContent className="text-sm text-muted-foreground p-4 border-x border-b rounded-b-md">
                  Yes, we provide RESTful APIs with comprehensive documentation, SDKs for popular languages (Python, JavaScript, Java),
                  webhook support for async processing, and batch processing capabilities for high-volume use cases.
                </AccordionContent>
              </AccordionItem>
              <AccordionItem value="item-8">
                <AccordionTrigger className="rounded-md border px-4 py-3 text-left">What about data privacy and security?</AccordionTrigger>
                <AccordionContent className="text-sm text-muted-foreground p-4 border-x border-b rounded-b-md">
                  All documents are encrypted in transit and at rest using AES-256. We offer on-premise deployment options,
                  GDPR compliance, and data retention policies. Documents are never stored longer than necessary for processing.
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
      </section>

      {/* Hidden Placeholder for Hero */}
      <div className="hidden" aria-hidden="true" />
    </main>
  )
}
