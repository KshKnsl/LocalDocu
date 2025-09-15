const DB_NAME = "aidocu_files_db";
const DB_VERSION = 1;
const STORE_NAME = "files";

type FileRecord = {
  key: string; // chatId/path/filename
  blob: Blob;
  mimeType?: string;
};

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: "key" });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function saveChatFileToLocal(chatId: string, filename: string, blob: Blob, mimeType?: string): Promise<string> {
  const db = await openDb();
  const key = `${chatId}/${filename}`;
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    const rec: FileRecord = { key, blob, mimeType };
    store.put(rec);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
  return URL.createObjectURL(blob);
}

export async function getChatFileLocalUrl(chatId: string, filename: string): Promise<string | undefined> {
  const db = await openDb();
  const key = `${chatId}/${filename}`;
  const rec: FileRecord | undefined = await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly");
    const store = tx.objectStore(STORE_NAME);
    const req = store.get(key);
    req.onsuccess = () => resolve(req.result as FileRecord | undefined);
    req.onerror = () => reject(req.error);
  });
  if (!rec) return undefined;
  return URL.createObjectURL(rec.blob);
}

export async function cloneChatFolderToLocal(chatId: string, files: { name: string; type?: string; key?: string }[]): Promise<Record<string, string>> {
  const result: Record<string, string> = {};
  for (const f of files) {
    try {
      let res: Response | undefined;
      if (f.key) {
        try {
          res = await fetch(`/api/document/download?key=${encodeURIComponent(f.key)}`);
        } catch (e) {
        }
      }
      if (!res) throw new Error("Missing key or proxy not reachable");
      const blob = await res.blob();
      const localUrl = await saveChatFileToLocal(chatId, f.name, blob, f.type);
      result[f.name] = localUrl;
    } catch (e) {
    }
  }
  return result;
}


