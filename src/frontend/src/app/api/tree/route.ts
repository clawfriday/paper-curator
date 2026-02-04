import { NextResponse } from "next/server";

// Extended timeout for large tree loading (can take 15+ seconds for large datasets)
export const maxDuration = 120; // 2 minutes

const BACKEND_URL = process.env.BACKEND_URL || "http://backend:8000";

export async function GET() {
  try {
    const res = await fetch(`${BACKEND_URL}/tree`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
      // Disable Next.js caching for this endpoint
      cache: "no-store",
    });

    if (!res.ok) {
      const errorText = await res.text();
      return NextResponse.json(
        { error: `Backend error: ${res.status}`, details: errorText },
        { status: res.status }
      );
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Failed to fetch tree:", error);
    return NextResponse.json(
      { error: "Failed to fetch tree", details: String(error) },
      { status: 500 }
    );
  }
}
