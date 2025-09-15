
import { NextRequest, NextResponse } from "next/server";
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import crypto from "crypto";
import path from "path";

const s3 = new S3Client({
  endpoint: process.env.FILEBASE_ENDPOINT || "https://s3.filebase.com",
  region: process.env.FILEBASE_REGION || "us-east-1",
  credentials: {
    accessKeyId: process.env.FILEBASE_ACCESS_KEY_ID!,
    secretAccessKey: process.env.FILEBASE_SECRET_ACCESS_KEY!,
  },
  forcePathStyle: true,
});
const bucket = process.env.FILEBASE_BUCKET_NAME!;


export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const file = formData.get("file");
  if (!(file instanceof File)) return null;
  const folder = (formData.get("folder") || "documents") as string;
  const key = `${folder.replace(/^\/+|\/+$/g, "").replace(/\.\./g, "")}/` +
    `${crypto.randomBytes(8).toString("hex")}-` +
    `${path.basename(file.name).replace(/\s+/g, "_") || "upload"}`;
  await s3.send(new PutObjectCommand({
    Bucket: bucket,
    Key: key,
    Body: Buffer.from(await file.arrayBuffer()),
    ContentType: file.type,
  }));
  return NextResponse.json({
    url: `/api/document/download?key=${encodeURIComponent(key)}`,
    filename: file.name,
    key,
  });
}

export const config = { api: { bodyParser: false } };
