import { NextResponse } from "next/server";

export const maxDuration = 300; // 5 minutes for classification

export async function POST() {
  const res = await fetch("http://backend:8000/papers/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
