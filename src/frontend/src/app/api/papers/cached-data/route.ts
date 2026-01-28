import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  const body = await request.json();
  const arxivId = body.arxiv_id;
  
  if (!arxivId) {
    return NextResponse.json({ error: "arxiv_id is required" }, { status: 400 });
  }
  
  const res = await fetch(`http://backend:8000/papers/${encodeURIComponent(arxivId)}/cached-data`, {
    method: "GET",
    headers: { "Content-Type": "application/json" },
  });
  
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
