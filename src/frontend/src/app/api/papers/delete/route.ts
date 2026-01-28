import { NextRequest, NextResponse } from "next/server";

export async function DELETE(request: NextRequest) {
  const body = await request.json();
  const arxivId = body.arxiv_id;
  
  if (!arxivId) {
    return NextResponse.json({ error: "arxiv_id is required" }, { status: 400 });
  }
  
  const res = await fetch(`http://backend:8000/papers/${encodeURIComponent(arxivId)}`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
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
}
