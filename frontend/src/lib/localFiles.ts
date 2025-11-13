const DB_NAME = "aidocu_files_db", DB_VERSION = 1, STORE_NAME = "files";
type FileRecord = { key: string; blob: Blob; mimeType?: string };

const openDb = (): Promise<IDBDatabase> => new Promise((resolve, reject) => {
  const req = indexedDB.open(DB_NAME, DB_VERSION);
  req.onupgradeneeded = () => { if (!req.result.objectStoreNames.contains(STORE_NAME)) req.result.createObjectStore(STORE_NAME, { keyPath: "key" }); };
  req.onsuccess = () => resolve(req.result);
  req.onerror = () => reject(req.error);
});

export const saveChatFileToLocal = async (chatId: string, filename: string, blob: Blob, mimeType?: string): Promise<string> => {
  const db = await openDb(), key = `${chatId}/${filename}`;
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).put({ key, blob, mimeType });
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
  return URL.createObjectURL(blob);
};

export const getChatFileLocalUrl = async (chatId: string, filename: string): Promise<string | undefined> => {
  const db = await openDb(), key = `${chatId}/${filename}`;
  const rec: FileRecord | undefined = await new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, "readonly"), req = tx.objectStore(STORE_NAME).get(key);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return rec ? URL.createObjectURL(rec.blob) : undefined;
};

export const cloneChatFolderToLocal = async (chatId: string, files: { name: string; type?: string; key?: string }[]): Promise<Record<string, string>> => {
  const result: Record<string, string> = {};
  for (const f of files) {
    try {
      if (!f.key) continue;
      const res = await fetch(`/api/document/download?key=${encodeURIComponent(f.key)}`);
      result[f.name] = await saveChatFileToLocal(chatId, f.name, await res.blob(), f.type);
    } catch {}
  }
  return result;
};


