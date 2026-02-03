import { NextRequest, NextResponse } from "next/server";

// Extended timeout for long-running summarization
export const maxDuration = 300; // 5 minutes

const BACKEND_URL = process.env.BACKEND_URL || "http://backend:8000";

export async function POST(request: NextRequest) {
  const body = await request.json();

  const response = await fetch(`${BACKEND_URL}/summarize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    return NextResponse.json(
      { error: `Backend error: ${response.status}` },
      { status: response.status }
    );
  }

  const data = await response.json();
  return NextResponse.json(data);
}
