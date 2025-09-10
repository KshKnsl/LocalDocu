import { NextRequest, NextResponse } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    await new Promise(resolve => setTimeout(resolve, 1500));
    const { message } = await req.json();
    if (!message) {
      return NextResponse.json(
        { error: 'No message provided' },
        { status: 400 }
      );
    }
    let response = "Dummy Response";

    return NextResponse.json({ response });
  } catch (error) {
    console.error('Error processing chat:', error);
    return NextResponse.json(
      { error: 'Error processing chat message' },
      { status: 500 }
    );
  }
}
