import { toast } from "sonner";

const trigger = (url: string, name?: string) => {
  const a = Object.assign(document.createElement("a"), { href: url, download: name || "" });
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
};

export const downloadBackendFile = async (os: "windows" | "mac" | "linux") => {
  try {
    const response = await fetch("https://api.github.com/repos/KshKnsl/LocalDocu/releases/latest");
    if (!response.ok) throw new Error("Failed to fetch release");
    const release = await response.json();
    const asset = release.assets.find((a: any) => a.name === "backend-unified.zip");
    if (!asset) throw new Error("Asset not found");
    trigger(asset.browser_download_url, asset.name);
    toast.success(`Unified backend downloaded`, { description: "Extract the zip and run the executable (works on any OS)" });
  } catch (error) {
    toast.error("Failed to download backend", { description: "Please try again later" });
    console.error(error);
  }
};

export const downloadJSON = (data: unknown, filename: string) => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  trigger(url, filename);
  URL.revokeObjectURL(url);
};
