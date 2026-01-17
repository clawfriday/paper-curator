#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="${CONFIG_FILE:-config/paperqa.yaml}"
IMAGE_NAME="$(awk -F': ' '$1=="service_container_name"{gsub(/"/,"",$2); print $2}' "${CONFIG_FILE}")"
PORT="$(awk -F': ' '$1=="service_base_url"{gsub(/"/,"",$2); print $2}' "${CONFIG_FILE}" | awk -F: '{print $3}' | tr -d /)"

# Build the extended GROBID image with FastAPI and nginx.
docker build --no-cache -t "${IMAGE_NAME}" -f docker/Dockerfile .
