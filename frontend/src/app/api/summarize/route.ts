import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    await new Promise((resolve) => setTimeout(resolve, 2000));

    const formData = await req.formData();
    const files: File[] = [];

    for (const entry of formData.entries()) {
      if (entry[1] instanceof File) {
        files.push(entry[1]);
      }
    }

    if (files.length === 0) {
      return NextResponse.json({ error: "No files provided" }, { status: 400 });
    }
    let summary = "Dummy Summary";

    return NextResponse.json({ summary });
  } catch (error) {
    console.error("Error summarizing document:", error);
    return NextResponse.json(
      { error: "Error summarizing document" },
      { status: 500 }
    );
  }
}
