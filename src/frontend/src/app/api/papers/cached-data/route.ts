import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_URL || "http://backend:8000";

export async function POST(request: NextRequest) {
  const body = await request.json();
  const arxivId = body.arxiv_id;
  
  if (!arxivId) {
    return NextResponse.json({ error: "arxiv_id is required" }, { status: 400 });
  }
  
  const res = await fetch(`${BACKEND_URL}/papers/${encodeURIComponent(arxivId)}/cached-data`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });
  
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
