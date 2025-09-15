import { NextRequest, NextResponse } from "next/server";
import { S3Client, GetObjectCommand } from "@aws-sdk/client-s3";

const s3 = new S3Client({
  region: process.env.FILEBASE_REGION || "us-east-1",
  endpoint: process.env.FILEBASE_ENDPOINT || "https://s3.filebase.com",
  credentials: {
    accessKeyId: process.env.FILEBASE_ACCESS_KEY_ID || "",
    secretAccessKey: process.env.FILEBASE_SECRET_ACCESS_KEY || "",
  },
  forcePathStyle: true,
});

export async function GET(req: NextRequest) {
  const key = req.nextUrl.searchParams.get("key");
  const bucket = process.env.FILEBASE_BUCKET_NAME;
  const out = await s3.send(new GetObjectCommand({ Bucket: bucket, Key: key||"" }));
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


