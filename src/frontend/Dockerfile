FROM node:20-slim AS base

# Install dependencies stage
FROM base AS deps
WORKDIR /app
# Copy both package.json and package-lock.json for reproducible builds
COPY package.json package-lock.json* ./
# Use npm ci for faster, reliable, reproducible builds (uses package-lock.json)
# Falls back to npm install if package-lock.json is missing
RUN npm ci || npm install

# Build stage
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# Some checkouts don't include /public; keep Docker build robust
RUN mkdir -p public && npm run build

# Production stage
FROM base AS runner
WORKDIR /app
ENV NODE_ENV=production

# Copy standalone output
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
