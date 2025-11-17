"use client";

import React, { useEffect } from "react";
import { redirect, useParams } from "next/navigation";
import ChatInterface from "@/components/chat/ChatInterface";
import { getChatById, addChat } from "@/lib/chatStorage";

export default function ChatPage() {
  const params = useParams();
  const chatId = params.chatId as string;

  useEffect(() => {
    if (!getChatById(chatId)) {
      redirect('/spaces');
    }
  }, [chatId]);

  return <ChatInterface initialChatId={chatId} />;
}
