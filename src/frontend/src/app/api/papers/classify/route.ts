import { NextResponse } from "next/server";
import { backendPost } from "@/lib/backend-proxy";

// Note: maxDuration is Vercel-only. Actual timeout is controlled by
// backendPost() which uses node:http (no undici headersTimeout limit).

export async function POST() {
  try {
    const { status, body } = await backendPost("/papers/classify", {
      timeoutMs: 900_000, // 15 minutes — classify 2000+ papers is slow
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
    console.error("Classify request failed:", error);
    return NextResponse.json(
      { error: "Classification request failed", details: String(error) },
      { status: 504 }
    );
  }
}
