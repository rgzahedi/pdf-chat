export const runtime = "nodejs";

import { Pinecone } from "@pinecone-database/pinecone";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { PineconeStore } from "@langchain/pinecone";
import { NextResponse } from "next/server";
import pRetry from "p-retry";

// Initialize Pinecone client
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY!,
});

export async function POST(req: Request) {
  try {
    // Get uploaded file
    const formData = await req.formData();
    const file = formData.get("file") as File;

    if (!file) {
      return new Response("No file provided", { status: 400 });
    }

    // Generate a unique document ID
    const documentId = crypto.randomUUID();

    // Convert file to blob
    const blob = new Blob([await file.arrayBuffer()], { type: file.type });

    // --- 1️⃣ Load and parse PDF ---
    const loader = new PDFLoader(blob);
    const docs = await loader.load();

    // --- 2️⃣ Split document into chunks ---
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const splitDocs = await splitter.splitDocuments(docs);

    // Add documentId to metadata
    const docsWithMetadata = splitDocs.map((doc) => ({
      ...doc,
      metadata: { ...doc.metadata, documentId },
    }));

    // --- 3️⃣ Summarize the first few chunks ---
    // Using ChatOpenAI for GPT-3.5-turbo
    const chatModel = new ChatOpenAI({
      model: "gpt-3.5-turbo",
      openAIApiKey: process.env.OPENAI_API_KEY!,
    });

    const firstFewChunks = splitDocs
      .slice(0, 3)
      .map((d) => d.pageContent)
      .join("\n\n");

    const summaryResponse = await chatModel.invoke([
      ["system", "You are a helpful AI assistant that summarizes documents."],
      ["user", `Summarize the following content:\n${firstFewChunks}`],
    ]);

    const summary =
      typeof summaryResponse.content === "string"
        ? summaryResponse.content
        : summaryResponse.content[0].text;

    // --- 4️⃣ Prepare embeddings ---
    const embeddings = new OpenAIEmbeddings({
      model: "text-embedding-3-small",
      openAIApiKey: process.env.OPENAI_API_KEY!,
      batchSize: 5, // helps avoid rate limit errors
    });

    const index = pinecone.Index(process.env.PINECONE_INDEX_NAME!);

    // --- 5️⃣ Upload embeddings to Pinecone with retry logic ---
    await pRetry(
      async () => {
        await PineconeStore.fromDocuments(docsWithMetadata, embeddings, {
          pineconeIndex: index,
        });
      },
      {
        retries: 3,
        onFailedAttempt: (err) => {
          console.warn(`Retrying upload (${err.attemptNumber}/3): ${err.message}`);
        },
      }
    );

    // --- ✅ 6️⃣ Return response ---
    return NextResponse.json({
      summary,
      documentId,
      pageCount: docs.length,
    });
  } catch (error) {
    console.error("Upload error:", error);
    const message = error instanceof Error ? error.message : "Unknown error";
    return new Response(message, { status: 500 });
  }
}
