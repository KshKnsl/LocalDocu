import { toast } from "sonner";

export const copyToClipboard = async (text: string, msg = "Copied to clipboard") => {
  try {
    await navigator.clipboard.writeText(text);
    toast.success(msg, { duration: 2000, position: "bottom-right" });
  } catch { toast.error("Failed to copy"); }
};

export const copyMessage = (content: string) => copyToClipboard(content, "Message copied to clipboard");
export const copyCurrentUrl = () => copyToClipboard(window.location.href, "Link copied to clipboard");
