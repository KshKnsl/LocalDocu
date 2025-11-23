import { FileWithUrl } from "@/components/ui/FileWithUrl";
import type { Citation } from "./api";


export interface MessageObject {
  message_id: string;
  author: "user" | "ai";
  content: string;
  created_at: string;
  files: FileWithUrl[];
  citations?: Citation[];
  model?: string;
}


export interface ChatDocument {
  chat_id: string;
  title: string;
  created_at: string;
  fileWithUrl: FileWithUrl[];
  message_objects: MessageObject[];
  description?: string;
  models?: string[];
}

const STORAGE_KEY = "aidocu_chats";

export function getAllChats(): ChatDocument[] {
  if (typeof window === "undefined") return [];
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) return parsed as ChatDocument[];
  } catch {}
  localStorage.removeItem(STORAGE_KEY);
  return [];
}


export function saveAllChats(chats: ChatDocument[]) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
  } catch {}
}

export function addChat(chat: ChatDocument) {
  if (!chat.chat_id || !chat.title || !chat.created_at) return;
  const chats = getAllChats();
  if (chats.find(c => c.chat_id === chat.chat_id)) {
    updateChat(chat);
    return;
  }
  chats.unshift(chat);
  saveAllChats(chats);
}

export function updateChat(chat: ChatDocument) {
  if (!chat.chat_id || !chat.title || !chat.created_at) return;
  const chats = getAllChats();
  const idx = chats.findIndex((c) => c.chat_id === chat.chat_id);
  if (idx !== -1) {
    chats[idx] = chat;
    saveAllChats(chats);
  } else {
    addChat(chat);
  }
}

export function deleteChat(chat_id: string) {
  if (!chat_id) return;
  const chats = getAllChats();
  const filteredChats = chats.filter((c) => c.chat_id !== chat_id);
  saveAllChats(filteredChats);
}

export function getChatById(chat_id: string): ChatDocument | undefined {
  if (!chat_id) return undefined;
  return getAllChats().find((c) => c.chat_id === chat_id);
}
