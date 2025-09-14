
import React from "react";
import ReactMarkdown from "react-markdown";
import rehypeRaw from "rehype-raw";

export function MarkdownRenderer({ content }: { content: string }) {
  return (
    <div className="prose max-w-none dark:prose-invert">
      <ReactMarkdown
        rehypePlugins={[rehypeRaw]}
        components={{
          a: ({ node, ...props }) => (
            <a {...props} target="_blank" rel="noopener noreferrer">
              {props.children}
            </a>
          ),
        }}
        skipHtml={false}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
