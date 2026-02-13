/**
 * Backend proxy using node:http for long-running requests.
 *
 * WHY THIS EXISTS:
 * Node.js built-in fetch (undici) enforces a hardcoded 300s headersTimeout.
 * This cannot be configured per-request. For backend operations that take
 * longer than 5 minutes (e.g. classify 2000+ papers, batch ingestion),
 * undici kills the connection with UND_ERR_HEADERS_TIMEOUT before the
 * backend finishes.
 *
 * The node:http module does NOT have this limitation — its `timeout` option
 * only fires on socket *inactivity*, not on time-to-first-header.
 *
 * NOTE: `export const maxDuration` is a Vercel-only feature and has NO
 * effect in standalone Node.js deployments (e.g. Singularity containers).
 */
import http from "node:http";

const BACKEND_URL = process.env.BACKEND_URL || "http://backend:8000";

interface ProxyResult {
  status: number;
  body: string;
}

/**
 * Proxy a POST request to the backend with configurable timeout.
 *
 * @param path   - Backend path (e.g. "/papers/classify")
 * @param opts.body      - Optional JSON string body
 * @param opts.timeoutMs - Socket inactivity timeout (default 10 min)
 */
export function backendPost(
  path: string,
  opts: { body?: string; timeoutMs?: number } = {}
): Promise<ProxyResult> {
  const { body, timeoutMs = 600_000 } = opts;

  return new Promise((resolve, reject) => {
    const url = new URL(path, BACKEND_URL);

    const req = http.request(
      {
        hostname: url.hostname,
        port: url.port || 80,
        path: url.pathname + url.search,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(body
            ? { "Content-Length": Buffer.byteLength(body).toString() }
            : {}),
        },
        timeout: timeoutMs,
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (chunk: Buffer) => chunks.push(chunk));
        res.on("end", () => {
          resolve({
            status: res.statusCode || 200,
            body: Buffer.concat(chunks).toString(),
          });
        });
      }
    );

    req.on("error", reject);
    req.on("timeout", () => {
      req.destroy();
      reject(
        new Error(`Backend request to ${path} timed out after ${timeoutMs / 1000}s`)
      );
    });

    if (body) {
      req.write(body);
    }
    req.end();
  });
}
