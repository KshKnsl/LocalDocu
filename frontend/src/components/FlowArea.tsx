"use client";


import React, { useState } from "react";
import mermaid from "mermaid";
import { Card } from "@/components/ui/card";
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from "@/components/ui/resizable";


export function FlowArea({
  chartData = `graph TD
    subgraph ClusterA
      A1-->A2
      A2-->A3
    end
    subgraph ClusterB
      B1-->B2
      B2-->B3
    end
    A3-->B1
    B3-->A1
    style ClusterA fill:#f9f,stroke:#333,stroke-width:2px
    style ClusterB fill:#bbf,stroke:#333,stroke-width:2px
    classDef important fill:#ff0,stroke:#333,stroke-width:2px;
    A2:::important
    B2:::important
  `
}: { chartData?: string }) {
  const [svg, setSvg] = useState<string>("");

  React.useEffect(() => {
    mermaid.initialize({ startOnLoad: false });
    mermaid.render("mermaid-svg", chartData)
      .then(({ svg }) => setSvg(svg))
      .catch(() => setSvg("<div class='text-red-500'>Invalid Mermaid syntax</div>"));
  }, [chartData]);

  return (
    <ResizablePanelGroup direction="vertical" style={{ width: '100%', minHeight: 200, minWidth: 320, maxWidth: 900, margin: '0 auto', background: 'var(--background)', borderRadius: 16, boxShadow: '0 2px 8px rgba(0,0,0,0.04)' }}>
      <ResizablePanel minSize={10} defaultSize={15} style={{ minHeight: 32, background: 'var(--background)', borderBottom: '1px solid var(--border)', borderTopLeftRadius: 16, borderTopRightRadius: 16 }}>
        <div className="flex items-center justify-center w-full select-none relative h-full">
          <div className="w-12 h-2 rounded-full bg-muted mb-1 shadow-sm" title="Resize" />
          <span className="font-semibold text-xs text-muted-foreground ml-2">Research Tools</span>
        </div>
      </ResizablePanel>
      <ResizableHandle withHandle />
      <ResizablePanel minSize={20} defaultSize={85} style={{ minHeight: 100, background: 'var(--background)', borderBottomLeftRadius: 16, borderBottomRightRadius: 16 }}>
        <Card className="p-4 flex flex-col items-center justify-center h-full shadow-none rounded-b-xl">
          <div className="text-base font-semibold mb-2">Research Paper Analysis</div>
          <div className="bg-muted/30 rounded p-2 w-full text-center text-muted-foreground" style={{ height: '100%', overflow: 'auto', maxHeight: 400 }}>
            <div dangerouslySetInnerHTML={{ __html: svg }} />
          </div>
        </Card>
      </ResizablePanel>
    </ResizablePanelGroup>
  );
}
