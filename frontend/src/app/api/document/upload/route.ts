import { NextRequest, NextResponse } from "next/server";
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import crypto from 'crypto';
import path from 'path';

const s3 = new S3Client({
  endpoint: process.env.FILEBASE_ENDPOINT,
  region: "us-east-1",
  credentials: {
    accessKeyId: process.env.FILEBASE_ACCESS_KEY_ID!,
    secretAccessKey: process.env.FILEBASE_SECRET_ACCESS_KEY!,
  },
  forcePathStyle: true
});

const bucket = process.env.FILEBASE_BUCKET_NAME!;

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get('file') as File;
    const contentType = formData.get('contentType') as string;
    
    if (!file) {
      return NextResponse.json(
        { error: 'No file provided' },
        { status: 400 }
      );
    }
    const fileExtension = path.extname(file.name);
    const randomId = crypto.randomBytes(16).toString('hex');
    const key = `documents/${randomId}${fileExtension}`;

    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);

    const command = new PutObjectCommand({
      Bucket: bucket,
      Key: key,
      Body: buffer,
      ContentType: contentType || file.type,
    });

    await s3.send(command);

    const url = `https://${bucket}.s3.filebase.com/${key}`;

    return NextResponse.json({
      url,
      filename: file.name,
      key
    });

  } catch (error) {
    console.error('Upload error:', error);
    return NextResponse.json(
      { error: 'Error uploading file to S3' },
      { status: 500 }
    );
  }
}

export const config = {
  api: {
    bodyParser: false,
  },
};
