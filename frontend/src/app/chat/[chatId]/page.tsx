"use client";

import React, { useEffect } from "react";
import { useParams } from "next/navigation";
import ChatInterface from "@/components/chat/ChatInterface";
import { getChatById, addChat } from "@/lib/chatStorage";

export default function ChatPage() {
  const params = useParams();
  const chatId = params.chatId as string;

  useEffect(() => {
    if (!getChatById(chatId)) {
      addChat({ chat_id: chatId, title: "New Chat", created_at: new Date().toISOString(), fileWithUrl: [], message_objects: [] });
    }
  }, [chatId]);

  return <ChatInterface initialChatId={chatId} />;
}
