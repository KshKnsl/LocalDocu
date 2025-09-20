
import { NextRequest, NextResponse } from "next/server";
import { PutObjectCommand } from "@aws-sdk/client-s3";
import crypto from "crypto";
import path from "path";
import { s3Config } from "@/lib/s3Config";


export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const file = formData.get("file");
  if (!(file instanceof File)) return null;
  const folder = (formData.get("folder") || "documents") as string;
  const key = `${folder.replace(/^\/+|\/+$/g, "").replace(/\.\./g, "")}/` +
    `${crypto.randomBytes(8).toString("hex")}-` +
    `${path.basename(file.name).replace(/\s+/g, "_") || "upload"}`;
  await s3Config.client.send(new PutObjectCommand({
    Bucket: s3Config.bucket,
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
