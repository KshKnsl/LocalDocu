import type { ChatDocument } from "./chatStorage";

export const getChatPreview = (chat: ChatDocument, max = 100) => {
  if (!chat.message_objects.length) return "No messages yet";
  const content = chat.message_objects[0].content.trim();
  return content.substring(0, max) + (content.length > max ? "..." : "");
};

export const getMessageCount = (chat: ChatDocument) => chat.message_objects.length;
export const getFileCount = (chat: ChatDocument) => (chat.fileWithUrl || []).length;

export const filterChatsByQuery = (chats: ChatDocument[], query: string) => {
  if (!query?.trim()) return chats;
  const q = query.toLowerCase();
  return chats.filter(chat => 
    chat.title.toLowerCase().includes(q) || 
    chat.message_objects.some(msg => msg.content.toLowerCase().includes(q))
  );
};

export const getChatStats = (chats: ChatDocument[]) => ({
  totalChats: chats.length,
  totalMessages: chats.reduce((s, c) => s + getMessageCount(c), 0),
  totalFiles: chats.reduce((s, c) => s + getFileCount(c), 0),
});
