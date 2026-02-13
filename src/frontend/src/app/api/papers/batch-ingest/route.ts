import { NextRequest, NextResponse } from "next/server";
import { backendPost } from "@/lib/backend-proxy";

// Note: maxDuration is Vercel-only. Actual timeout is controlled by
// backendPost() which uses node:http (no undici headersTimeout limit).

export async function POST(request: NextRequest) {
  try {
    const reqBody = await request.json();

    const { status, body } = await backendPost("/papers/batch-ingest", {
      body: JSON.stringify(reqBody),
      timeoutMs: 1_800_000, // 30 minutes — batch ingestion can be very slow
    });

    try {
      const data = JSON.parse(body);
      return NextResponse.json(data, { status });
    } catch {
      return NextResponse.json(
        { error: "Invalid response from backend", details: body.slice(0, 500) },
        { status: 502 }
      );
    }
  } catch (error) {
    console.error("Batch ingest request failed:", error);
    return NextResponse.json(
      { error: "Batch ingest request failed", details: String(error) },
      { status: 504 }
    );
  }
}
