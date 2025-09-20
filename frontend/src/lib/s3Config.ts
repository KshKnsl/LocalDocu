import { S3Client } from "@aws-sdk/client-s3";
export const s3Client = new S3Client({
  endpoint: process.env.FILEBASE_ENDPOINT || "https://s3.filebase.com",
  region: process.env.FILEBASE_REGION || "us-east-1",
  credentials: {
    accessKeyId: process.env.FILEBASE_ACCESS_KEY_ID!,
    secretAccessKey: process.env.FILEBASE_SECRET_ACCESS_KEY!,
  },
  forcePathStyle: true,
});

export const s3Config = {
  bucket: process.env.FILEBASE_BUCKET_NAME!,
  client: s3Client,
};