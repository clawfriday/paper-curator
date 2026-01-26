import { NextRequest, NextResponse } from "next/server";

// Extended timeout for long-running QA (PaperQA2 can take a while)
export const maxDuration = 300; // 5 minutes

export async function POST(request: NextRequest) {
  const body = await request.json();

  const response = await fetch("http://backend:8000/qa", {
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
