import { NextRequest, NextResponse } from "next/server";
import { backendPost } from "@/lib/backend-proxy";

// Topic query runs multi-paper RAG (per-paper evidence + LLM aggregation),
// which can take 5-10+ minutes depending on pool size. The default undici
// fetch proxy would kill the connection at 300s (UND_ERR_HEADERS_TIMEOUT).

export async function POST(
  request: NextRequest,
  { params }: { params: { topicId: string } }
) {
  try {
    const reqBody = await request.json();

    const { status, body } = await backendPost(
      `/topic/${params.topicId}/query`,
      {
        body: JSON.stringify(reqBody),
        timeoutMs: 900_000, // 15 minutes — multi-paper RAG is slow
      }
    );

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
    console.error(`Topic query failed for topic ${params.topicId}:`, error);
    return NextResponse.json(
      { error: "Topic query request failed", details: String(error) },
      { status: 504 }
    );
  }
}
