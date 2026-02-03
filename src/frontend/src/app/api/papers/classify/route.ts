import { NextResponse } from "next/server";

export const maxDuration = 300; // 5 minutes for classification

const BACKEND_URL = process.env.BACKEND_URL || "http://backend:8000";

export async function POST() {
  const res = await fetch(`${BACKEND_URL}/papers/classify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  
  const data = await res.json();
  return NextResponse.json(data, { status: res.status });
}
