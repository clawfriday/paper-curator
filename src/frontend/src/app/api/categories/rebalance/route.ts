import { NextResponse } from "next/server";
import { backendPost } from "@/lib/backend-proxy";

// Note: maxDuration is Vercel-only. Actual timeout is controlled by
// backendPost() which uses node:http (no undici headersTimeout limit).

export async function POST() {
  try {
    const { status, body } = await backendPost("/categories/rebalance", {
      timeoutMs: 900_000, // 15 minutes — same as classify
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
    console.error("Rebalance request failed:", error);
    return NextResponse.json(
      { error: "Rebalance request failed", details: String(error) },
      { status: 504 }
    );
  }
}
