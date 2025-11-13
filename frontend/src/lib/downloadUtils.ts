import { toast } from "sonner";

const trigger = (url: string, name?: string) => {
  const a = Object.assign(document.createElement("a"), { href: url, download: name || "" });
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
};

export const downloadBackendFile = (os: "windows" | "mac" | "linux") => {
  const files = { windows: "/backend-windows.py", mac: "/backend-mac.py", linux: "/backend-ubuntu.py" };
  trigger(files[os], `backend-${os}.py`);
  toast.success(`Backend downloaded for ${os}`, { description: "See file for setup instructions" });
};

export const downloadJSON = (data: unknown, filename: string) => {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  trigger(url, filename);
  URL.revokeObjectURL(url);
};
