
import { NextRequest, NextResponse } from "next/server";
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import crypto from "crypto";
import path from "path";

const s3 = new S3Client({
  endpoint: process.env.FILEBASE_ENDPOINT,
  region: "us-east-1",
  credentials: {
    accessKeyId: process.env.FILEBASE_ACCESS_KEY_ID!,
    secretAccessKey: process.env.FILEBASE_SECRET_ACCESS_KEY!,
  },
  forcePathStyle: true,
});
const bucket = process.env.FILEBASE_BUCKET_NAME!;


export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("file");
    if (!(file instanceof File)) return null;
    const ext = path.extname(file.name);
    const key = `documents/${crypto.randomBytes(16).toString("hex")}${ext}`;
    const buffer = Buffer.from(await file.arrayBuffer());

    await s3.send(new PutObjectCommand({
      Bucket: bucket,
      Key: key,
      Body: buffer,
      ContentType: file.type,
    }));

    return NextResponse.json({
      url: `https://${bucket}.s3.filebase.com/${key}`,
      filename: file.name,
      key,
    });
  } catch (error) {
    return NextResponse.json({ error: "Error uploading file to S3" }, { status: 500 });
  }
}

export const config = { api: { bodyParser: false } };
