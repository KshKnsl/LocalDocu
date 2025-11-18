"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Download, Server, Check, X, Loader2 } from "lucide-react"
import { toast } from "sonner"
import { downloadBackendFile } from "@/lib/downloadUtils"

const DEFAULT_BACKEND = "https://minor-project-6v6z.vercel.app/api"

export function BackendConfig() {
  const [backendMode, setBackendMode] = useState<"default" | "custom">("default")
  const [customUrl, setCustomUrl] = useState("http://localhost:8000")
  const [isTestingConnection, setIsTestingConnection] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState<"idle" | "success" | "error">("idle")

  useEffect(() => {
    const saved = localStorage.getItem("backendUrl")
    const savedMode = localStorage.getItem("backendMode") as "default" | "custom" | null
    
    if (savedMode) {
      setBackendMode(savedMode)
      if (savedMode === "custom" && saved) {
        setCustomUrl(saved)
      }
    }
  }, [])

  const testConnection = async (url: string) => {
    setIsTestingConnection(true)
    try {
      const res = await fetch(`${url}/health`, { method: "GET", signal: AbortSignal.timeout(5000) })
      const success = res.ok
      setConnectionStatus(success ? "success" : "error")
      toast[success ? "success" : "error"](success ? "Backend connected!" : "Backend not ready")
      return success
    } catch {
      setConnectionStatus("error")
      toast.error("Failed to connect", { description: "Check if backend is running" })
      return false
    } finally {
      setIsTestingConnection(false)
    }
  }

  const saveBackendConfig = async () => {
    const urlToSave = backendMode === "custom" ? customUrl : DEFAULT_BACKEND
    if (backendMode === "custom" && !customUrl) return toast.error("Please enter a backend URL")

    await testConnection(urlToSave)

    localStorage.setItem("backendUrl", urlToSave)
    localStorage.setItem("backendMode", backendMode)
    if (typeof window !== "undefined") (window as any).NEXT_PUBLIC_BACKEND_URL = urlToSave
    toast.success("Configuration saved!", { description: "Page will reload" })
    setTimeout(() => window.location.reload(), 1500)
  }



  const currentBackendUrl = backendMode === "custom" ? customUrl : DEFAULT_BACKEND

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Server className="w-5 h-5" />
          Backend Configuration
        </CardTitle>
        <CardDescription>
          Choose to use our backend or host your own private instance
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Current Status */}
        <div className="flex items-center justify-between p-3 rounded-lg border bg-muted/50">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${
              connectionStatus === "success" ? "bg-green-500" : 
              connectionStatus === "error" ? "bg-red-500" : 
              "bg-gray-400"
            }`} />
            <span className="text-sm font-medium">
              Current Backend: {backendMode === "default" ? "Our Server" : "Custom"}
            </span>
          </div>
          <Button 
            size="sm" 
            variant="outline"
            onClick={() => testConnection(currentBackendUrl)}
            disabled={isTestingConnection}
          >
            {isTestingConnection ? (
              <><Loader2 className="w-3 h-3 mr-1 animate-spin" /> Testing...</>
            ) : (
              <>Test Connection</>
            )}
          </Button>
        </div>

        {/* Backend Mode Selection */}
        <div className="space-y-3">
          <label className="text-sm font-medium">Backend Mode</label>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {/* Default Backend Option */}
            <Card 
              className={`cursor-pointer transition-all ${
                backendMode === "default" 
                  ? "border-primary ring-2 ring-primary/20" 
                  : "hover:border-primary/50"
              }`}
              onClick={() => setBackendMode("default")}
            >
              <CardContent className="p-4">
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <h4 className="font-semibold text-sm">Our Backend</h4>
                      <Badge variant="secondary" className="text-xs">Recommended</Badge>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Use our managed backend (still processes locally via Ollama)
                    </p>
                  </div>
                  {backendMode === "default" && (
                    <Check className="w-5 h-5 text-primary" />
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Custom Backend Option */}
            <Card 
              className={`cursor-pointer transition-all ${
                backendMode === "custom" 
                  ? "border-primary ring-2 ring-primary/20" 
                  : "hover:border-primary/50"
              }`}
              onClick={() => setBackendMode("custom")}
            >
              <CardContent className="p-4">
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <h4 className="font-semibold text-sm">Self-Hosted</h4>
                      <Badge variant="outline" className="text-xs">100% Private</Badge>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Host your own backend for complete control
                    </p>
                  </div>
                  {backendMode === "custom" && (
                    <Check className="w-5 h-5 text-primary" />
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Custom URL Input */}
        {backendMode === "custom" && (
          <div className="space-y-2 animate-in slide-in-from-top-2">
            <label className="text-sm font-medium">Your Backend URL</label>
            <Input
              type="url"
              placeholder="http://localhost:8000 or https://your-ngrok-url.app"
              value={customUrl}
              onChange={(e) => setCustomUrl(e.target.value)}
              className="font-mono text-sm"
            />
            <p className="text-xs text-muted-foreground">
              Enter the URL where your backend is running (including http:// or https://)
            </p>
          </div>
        )}

        {/* Download Backend Section */}
        {backendMode === "custom" && (
          <div className="space-y-3 pt-3 border-t animate-in slide-in-from-top-3">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-semibold">Download Backend</h4>
                <p className="text-xs text-muted-foreground">Get the Python backend to run locally</p>
              </div>
            </div>
            
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
              <Button 
                size="sm" 
                variant="outline"
                onClick={() => downloadBackendFile("windows")}
                className="flex items-center gap-2"
              >
                <Download className="w-3 h-3" />
                Windows
              </Button>
              <Button 
                size="sm" 
                variant="outline"
                onClick={() => downloadBackendFile("mac")}
                className="flex items-center gap-2"
              >
                <Download className="w-3 h-3" />
                macOS
              </Button>
              <Button 
                size="sm" 
                variant="outline"
                onClick={() => downloadBackendFile("linux")}
                className="flex items-center gap-2"
              >
                <Download className="w-3 h-3" />
                Linux
              </Button>
            </div>

            <div className="p-3 rounded-lg bg-muted text-xs space-y-2">
              <p className="font-medium">Quick Setup:</p>
              <ol className="list-decimal list-inside space-y-1 text-muted-foreground">
                <li>Download the backend file for your OS above</li>
                <li>Install dependencies: <code className="bg-background px-1 py-0.5 rounded">pip install fastapi uvicorn langchain langchain-community langchain-ollama langchain-google-genai langchain-huggingface langchain-chroma langchain-text-splitters pymupdf pyngrok chromadb requests</code></li>
                <li>Run: <code className="bg-background px-1 py-0.5 rounded">python backend-windows.py</code> (or mac/ubuntu)</li>
                <li>Follow the on-screen instructions to set up ngrok (optional)</li>
                <li>Copy the URL and paste it above, then save</li>
              </ol>
              <p className="mt-2 text-muted-foreground">Complete instructions are in the downloaded Python file!</p>
            </div>
          </div>
        )}

        {/* Save Button */}
        <div className="flex items-center gap-2 pt-2">
          <Button 
            onClick={saveBackendConfig}
            disabled={isTestingConnection}
            className="flex-1"
          >
            {isTestingConnection ? (
              <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> Testing...</>
            ) : (
              <>Save Configuration</>
            )}
          </Button>
          
          {connectionStatus === "success" && (
            <div className="flex items-center gap-1 text-sm text-green-600">
              <Check className="w-4 h-4" />
              <span>Connected</span>
            </div>
          )}
          {connectionStatus === "error" && (
            <div className="flex items-center gap-1 text-sm text-red-600">
              <X className="w-4 h-4" />
              <span>Error</span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

export function BackendConfigDialog({ collapsed = false }: { collapsed?: boolean }) {
  const [status, setStatus] = useState<"idle" | "success" | "error">("idle")

  useEffect(() => {
    const checkStatus = async () => {
      const savedUrl = localStorage.getItem("backendUrl")
      const savedMode = localStorage.getItem("backendMode")
      const checkUrl = savedMode === "custom" ? savedUrl : DEFAULT_BACKEND
      if (!checkUrl) return
      
      try {
        const res = await fetch(`${checkUrl}/health`, { method: "GET", signal: AbortSignal.timeout(5000) })
        setStatus(res.ok ? "success" : "error")
      } catch {
        setStatus("error")
      }
    }

    checkStatus()
    const interval = setInterval(checkStatus, 30000)
    return () => clearInterval(interval)
  }, [])

  const statusColor = status === "success" ? "bg-green-500" : status === "error" ? "bg-red-500" : "bg-gray-400"
  const statusText = status === "success" ? "Connected" : status === "error" ? "Offline" : "Checking..."

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm" className={`w-full gap-2 justify-start ${collapsed ? 'px-2' : ''}`}>
          <div className={`w-2 h-2 rounded-full ${statusColor}`} />
          {!collapsed && <Server className="w-4 h-4" />}
          {!collapsed && (
            <>
              <span className="flex-1 text-left">Backend Settings</span>
              <span className="text-xs text-muted-foreground">{statusText}</span>
            </>
          )}
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-[95vw] sm:max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Backend Configuration</DialogTitle>
        </DialogHeader>
        <BackendConfig />
      </DialogContent>
    </Dialog>
  )
}
