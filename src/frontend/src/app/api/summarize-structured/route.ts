import { NextRequest, NextResponse } from "next/server";

// Extended timeout for structured summarization (can take several minutes)
export const maxDuration = 600;

export async function POST(request: NextRequest) {
  const body = await request.json();
  
  const response = await fetch("http://backend:8000/summarize/structured", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const errorText = await response.text();
    return NextResponse.json(
      { error: `Backend error: ${response.status}`, details: errorText },
      { status: response.status }
    );
  }

  const data = await response.json();
  return NextResponse.json(data);
}
