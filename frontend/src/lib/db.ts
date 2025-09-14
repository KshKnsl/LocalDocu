import { Pool } from 'pg';
const connectionString = process.env.DATABASE_URL || '';

export const pool = new Pool({
  connectionString,
  ssl: { rejectUnauthorized: false },
});

export async function createChat(userId: string, title: string, preview: string) {
  const res = await pool.query(
    `INSERT INTO chats (user_id, title, preview, created_at) VALUES ($1, $2, $3, NOW()) RETURNING *`,
    [userId, title, preview]
  );
  return res.rows[0];
}

export async function getChats(userId: string) {
  const res = await pool.query(
    `SELECT id, title, preview, created_at FROM chats WHERE user_id = $1 ORDER BY created_at DESC`,
    [userId]
  );
  return res.rows;
}

// --- Message Functions ---
export async function addMessage(chatId: string, senderType: 'user' | 'bot', content: string, timestamp: Date) {
  const res = await pool.query(
    `INSERT INTO messages (chat_id, sender_type, content, timestamp) VALUES ($1, $2, $3, $4) RETURNING *`,
    [chatId, senderType, content, timestamp]
  );
  return res.rows[0];
}

export async function getMessages(chatId: string) {
  const res = await pool.query(
    `SELECT id, sender_type, content, timestamp FROM messages WHERE chat_id = $1 ORDER BY timestamp ASC`,
    [chatId]
  );
  return res.rows;
}

// --- File Functions ---
export async function addFile(messageId: string, url: string, filename: string) {
  const res = await pool.query(
    `INSERT INTO files (message_id, url, filename, uploaded_at) VALUES ($1, $2, $3, NOW()) RETURNING *`,
    [messageId, url, filename]
  );
  return res.rows[0];
}

export async function getFiles(messageId: string) {
  const res = await pool.query(
    `SELECT id, url, filename, uploaded_at FROM files WHERE message_id = $1 ORDER BY uploaded_at ASC`,
    [messageId]
  );
  return res.rows;
}
