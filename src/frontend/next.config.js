/** @type {import('next').NextConfig} */

// Backend URL: defaults to Docker service name, override with BACKEND_URL env var
// For Docker: http://backend:8000 (default)
// For Singularity/HPC: http://localhost:3100
const backendUrl = process.env.BACKEND_URL || "http://backend:8000";

const nextConfig = {
  output: "standalone",
  // Expose BACKEND_URL to API routes at runtime
  serverRuntimeConfig: {
    backendUrl: backendUrl,
  },
  // Also expose for build-time rewrites
  env: {
    BACKEND_URL: backendUrl,
  },
  async rewrites() {
    return [
      // Core endpoints
      {
        source: "/api/arxiv/:path*",
        destination: `${backendUrl}/arxiv/:path*`,
      },
      {
        source: "/api/pdf/:path*",
        destination: `${backendUrl}/pdf/:path*`,
      },
      {
        source: "/api/classify",
        destination: `${backendUrl}/classify`,
      },
      {
        source: "/api/abbreviate",
        destination: `${backendUrl}/abbreviate`,
      },
      {
        source: "/api/embed",
        destination: `${backendUrl}/embed`,
      },
      // Note: /api/qa is handled by custom API route with extended timeout
      {
        source: "/api/health",
        destination: `${backendUrl}/health`,
      },
      {
        source: "/api/config",
        destination: `${backendUrl}/config`,
      },
      // Database & Tree endpoints
      {
        source: "/api/papers/:path*",
        destination: `${backendUrl}/papers/:path*`,
      },
      {
        source: "/api/tree",
        destination: `${backendUrl}/tree`,
      },
      {
        source: "/api/tree/:path*",
        destination: `${backendUrl}/tree/:path*`,
      },
      // Feature endpoints
      {
        source: "/api/repos/:path*",
        destination: `${backendUrl}/repos/:path*`,
      },
      {
        source: "/api/references/:path*",
        destination: `${backendUrl}/references/:path*`,
      },
      // Note: /api/summarize is handled by custom API route with extended timeout
    ];
  },
};

module.exports = nextConfig;
