import { NextRequest, NextResponse } from "next/server";

export async function GET(req: NextRequest) {
  const key = req.nextUrl.searchParams.get("key");
  if (key?.startsWith("local-")) {
    return new NextResponse("File is stored locally in browser. Use local download functionality.", {
      status: 410, // Gone - resource is no longer available at this location
      headers: {
        "Content-Type": "text/plain",
      },
    });
  }

  return new NextResponse("Download functionality moved to local browser storage.", {
    status: 410,
    headers: {
      "Content-Type": "text/plain",
    },
  });
}


