#!/usr/bin/env bash
# Launch/stop/status Paper Curator on host Docker daemon from inside openclaw container.
#
# Usage:
#   ./scripts/host-services.sh start|stop|status|logs

set -euo pipefail

ACTION="${1:-status}"

CONTAINER_DB="paper-curator-db"
CONTAINER_BE="paper-curator-backend"
CONTAINER_FE="paper-curator-frontend"

# Host path where this workspace is mounted for host Docker
WORKSPACE_HOST_PATH="${WORKSPACE_HOST_PATH:-/workspace}"
PROJECT_HOST="${PROJECT_HOST:-${WORKSPACE_HOST_PATH}/repos/paper-curator}"
PROJECT_SRC_CONTAINER="/workspace/repos/paper-curator"

BACKEND_IMAGE="paper-curator-backend:local"
FRONTEND_IMAGE="paper-curator-frontend:local"

DB_PORT="${DB_PORT:-5432}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

DB_USER="${POSTGRES_USER:-curator}"
DB_PASS="${POSTGRES_PASSWORD:-curator}"
DB_NAME="${POSTGRES_DB:-paper_curator}"

need_docker() {
  if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Cannot connect to Docker daemon. Is /var/run/docker.sock mounted?"
    exit 1
  fi
}

wait_http() {
  local url="$1"
  local name="$2"
  local max_wait="${3:-60}"
  local i=0
  echo "Waiting for ${name} (${url})..."
  until curl -fsS "$url" >/dev/null 2>&1; do
    i=$((i+1))
    if [ "$i" -ge "$max_wait" ]; then
      echo "ERROR: ${name} not ready within ${max_wait}s"
      return 1
    fi
    sleep 1
  done
  echo "${name} is ready"
}

build_images() {
  echo "Building backend image..."
  docker build -t "$BACKEND_IMAGE" "$PROJECT_HOST/src/backend"
  echo "Building frontend image..."
  docker build -t "$FRONTEND_IMAGE" "$PROJECT_HOST/src/frontend"
}

start_db() {
  docker rm -f "$CONTAINER_DB" >/dev/null 2>&1 || true
  docker run -d \
    --name "$CONTAINER_DB" \
    --network host \
    -e POSTGRES_USER="$DB_USER" \
    -e POSTGRES_PASSWORD="$DB_PASS" \
    -e POSTGRES_DB="$DB_NAME" \
    -v "$PROJECT_HOST/storage/pgdata:/var/lib/postgresql/data" \
    -v "$PROJECT_HOST/src/backend/init.sql:/docker-entrypoint-initdb.d/init.sql:ro" \
    pgvector/pgvector:pg16 >/dev/null
}

start_backend() {
  docker rm -f "$CONTAINER_BE" >/dev/null 2>&1 || true
  docker run -d \
    --name "$CONTAINER_BE" \
    --network host \
    -e PYTHONUNBUFFERED=1 \
    -e DATABASE_URL="postgresql://${DB_USER}:${DB_PASS}@127.0.0.1:${DB_PORT}/${DB_NAME}" \
    -v "$PROJECT_HOST/storage:/app/storage" \
    -v "$PROJECT_HOST/config:/app/config" \
    -v "$HOME:/host_home:ro" \
    "$BACKEND_IMAGE" >/dev/null
}

start_frontend() {
  docker rm -f "$CONTAINER_FE" >/dev/null 2>&1 || true
  docker run -d \
    --name "$CONTAINER_FE" \
    --network host \
    -e BACKEND_URL="http://127.0.0.1:${BACKEND_PORT}" \
    "$FRONTEND_IMAGE" >/dev/null
}

stop_all() {
  docker rm -f "$CONTAINER_FE" "$CONTAINER_BE" "$CONTAINER_DB" >/dev/null 2>&1 || true
}

status_all() {
  docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep -E "(NAMES|paper-curator-)" || true
  echo "Backend health:"; curl -fsS "http://127.0.0.1:${BACKEND_PORT}/health" || true; echo
  echo "Frontend URL: http://127.0.0.1:${FRONTEND_PORT}"
}

logs_all() {
  docker logs --tail 100 "$CONTAINER_DB" || true
  docker logs --tail 100 "$CONTAINER_BE" || true
  docker logs --tail 100 "$CONTAINER_FE" || true
}

need_docker

# If host mirror repo is missing source files, sync from container workspace.
if [ ! -d "$PROJECT_HOST/src/backend" ] || [ ! -d "$PROJECT_HOST/src/frontend" ]; then
  echo "Host project mirror incomplete at $PROJECT_HOST, syncing from $PROJECT_SRC_CONTAINER ..."
  mkdir -p "$PROJECT_HOST"
  rm -rf "$PROJECT_HOST"/*
  cp -a "$PROJECT_SRC_CONTAINER"/. "$PROJECT_HOST"/
fi

mkdir -p "$PROJECT_HOST/storage/pgdata"

case "$ACTION" in
  start)
    build_images
    start_db
    sleep 4
    start_backend
    wait_http "http://127.0.0.1:${BACKEND_PORT}/health" "Backend" 120
    start_frontend
    wait_http "http://127.0.0.1:${FRONTEND_PORT}" "Frontend" 120
    status_all
    ;;
  stop)
    stop_all
    ;;
  status)
    status_all
    ;;
  logs)
    logs_all
    ;;
  *)
    echo "Usage: $0 {start|stop|status|logs}"
    exit 1
    ;;
esac
