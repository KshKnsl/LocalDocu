import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
    const { url } = await req.json();

    if (!url) {
      return NextResponse.json(
        { success: false, message: "No document URL provided" },
        { status: 400 }
      );
    }

    return NextResponse.json({
      success: true,
      message: "Document processed and embeddings created successfully",
      results: {
        documentId: "doc_" + url.split("/").pop(),
        status: "embeddings_created",
        chunkCount: 3,
        summary: "This is an initial summary of the uploaded document."
      }
    });
  }
