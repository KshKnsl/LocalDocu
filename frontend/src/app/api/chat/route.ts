import { NextResponse } from 'next/server';

export async function POST() {
  return NextResponse.json({
    response: "I understand you're asking about {topic}. Let me help you with that."
  });
}
