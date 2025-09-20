import { NextRequest, NextResponse } from "next/server";
import { GetObjectCommand } from "@aws-sdk/client-s3";
import { s3Config } from "@/lib/s3Config";

export async function GET(req: NextRequest) {
  const key = req.nextUrl.searchParams.get("key");
  const out = await s3Config.client.send(new GetObjectCommand({ Bucket: s3Config.bucket, Key: key||"" }));
  const filename = key?.split("/").pop() || "file";
  return new NextResponse(out.Body as any, {
    status: 200,
    headers: {
      "Content-Type": out.ContentType || "application/octet-stream",
      "Cache-Control": "private, no-store",
      "Content-Disposition": `inline; filename="${filename}"`,
    },
  });
}


