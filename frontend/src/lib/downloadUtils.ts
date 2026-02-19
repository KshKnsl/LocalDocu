import { toast } from "sonner";

const trigger = (url: string, name?: string) => {
  const a = Object.assign(document.createElement("a"), { href: url, download: name || "" });
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
};

export const copyInstallCommand = async (os: "windows" | "mac" | "linux") => {
  const cmds: Record<string, string> = {
    windows: `iwr -useb https://localdocu.vercel.app/install.ps1 | iex`,
    mac: `curl -fsSL https://localdocu.vercel.app/install.sh | bash`,
    linux: `curl -fsSL https://localdocu.vercel.app/install.sh | bash`,
  };
  const cmd = cmds[os];
    await navigator.clipboard.writeText(cmd);
    toast.success("Install command copied to clipboard", { description: cmd });
};

export const downloadJSON = (data: unknown, filename: string) => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  trigger(url, filename);
  URL.revokeObjectURL(url);
};
